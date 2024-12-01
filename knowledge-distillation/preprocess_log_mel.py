# preprocess_features.py
import os
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch
from transformers import WhisperFeatureExtractor
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import soundfile as sf

logger = get_logger(__name__)

class PreprocessedWhisperDataset(Dataset):
    """
    Dataset for loading preprocessed Whisper features
    """
    def __init__(self, features_dir, audio_fpaths, sampling_rate=16000):
        self.features_dir = features_dir
        self.audio_fpaths = audio_fpaths
        self.sampling_rate = sampling_rate
        
    def __getitem__(self, idx):
        feature_path = os.path.join(
            self.features_dir,
            os.path.basename(self.audio_fpaths[idx]).replace('.flac', '.h5')
        )
        
        with h5py.File(feature_path, 'r') as f:
            return {
                'input_features': f['input_features'][:],
                'batch_id': idx,
                'whisper_transcript': f['whisper_transcript'][()].decode('utf-8'),
                'last_segment_transcript': f['last_segment_transcript'][()].decode('utf-8'),
                'condition_on_prev': f['condition_on_prev'][()].decode('utf-8')
            }
    
    def __len__(self):
        return len(self.audio_fpaths)
    
def preprocess_dataset(
    audio_fpaths,
    output_dir,
    feature_extractor,
    accelerator,
    batch_size=32,
    sampling_rate=16000
):
    """
    Process audio files using accelerator for multi-GPU support
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_extractor = accelerator.prepare(feature_extractor)
    device = accelerator.device
    
    total_processes = accelerator.num_processes
    process_idx = accelerator.process_index
    
    per_process_paths = audio_fpaths[process_idx::total_processes]
    
    for i in tqdm(
        range(0, len(per_process_paths), batch_size),
        desc=f'Process {process_idx} extracting features',
        disable=not accelerator.is_local_main_process
    ):
        batch_paths = per_process_paths[i:i + batch_size]
        
        audio_batch = []
        metadata_batch = []
        
        for path in batch_paths:
            # Load audio data as numpy first
            audio_data = sf.read(path)[0]
            audio_batch.append(audio_data)
            
            # Load transcripts
            txt_path = path.replace('.flac', '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                metadata_batch.append({
                    'transcript': lines[0].strip(),
                    'last_segment': lines[2].strip(),
                    'prev': lines[4].strip()
                })
        
        # Process features
        with accelerator.autocast():
            features = feature_extractor(
                audio_batch,
                sampling_rate=sampling_rate
            )
        
        # Save features
        for path, feature, metadata in zip(
            batch_paths, 
            features.input_features, 
            metadata_batch
        ):
            save_path = os.path.join(
                output_dir,
                os.path.basename(path).replace('.flac', '.h5')
            )
            
            with h5py.File(save_path, 'w') as f:
                # Save feature directly as it's already numpy array
                f.create_dataset('input_features', data=feature)
                f.create_dataset('whisper_transcript', data=metadata['transcript'].encode('utf-8'))
                f.create_dataset('last_segment_transcript', data=metadata['last_segment'].encode('utf-8'))
                f.create_dataset('condition_on_prev', data=metadata['prev'].encode('utf-8'))
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    import argparse
    import os.path as osp
    
    parser = argparse.ArgumentParser(description="Preprocess Whisper features with multi-GPU support")
    
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="Path to training dataset manifest"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Audio files root directory"
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="openai/whisper-large-v2",
        help="Whisper model identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/home/ntuspeechlabtaipei1/forbes/log_mel",
        help="Feature output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Processing batch size"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Custom dataset name"
    )
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output directory
    if args.dataset_name is None:
        base_name = osp.splitext(osp.basename(args.train_manifest))[0]
        output_dir = f"{args.output_dir}/{base_name}"     
    else:
        output_dir = f"{args.output_dir}/{args.dataset_name}"
    
    # Load audio paths
    def load_audio_fpaths(manifest_fpath, root=None):
        audio_fpaths = []
        with open(manifest_fpath, "r") as fr:
            if root is None:
                root = fr.readline().strip()
            else:
                _ = fr.readline()
            for line in fr:
                audio_fpath = osp.join(root, line.strip())
                audio_fpaths.append(audio_fpath)
        return audio_fpaths
    
    # Initialize feature extractor
    if accelerator.is_local_main_process:
        logger.info(f"Loading feature extractor: {args.model_card}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_card)
    
    # Load audio paths
    if accelerator.is_local_main_process:
        logger.info(f"Reading manifest: {args.train_manifest}")
    audio_fpaths = load_audio_fpaths(args.train_manifest, root=args.root)
    if accelerator.is_local_main_process:
        logger.info(f"Found {len(audio_fpaths)} audio files")
    
    # Run preprocessing
    if accelerator.is_local_main_process:
        logger.info("Starting feature extraction...")
    
    preprocess_dataset(
        audio_fpaths=audio_fpaths,
        output_dir=output_dir,
        feature_extractor=feature_extractor,
        accelerator=accelerator,
        batch_size=args.batch_size
    )
    
    if accelerator.is_local_main_process:
        logger.info(f"Feature extraction completed! Features saved to: {output_dir}")
    
    
    
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 preprocess_log_mel.py \
#     --train_manifest /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/train_0.6_20241130_175240.tsv \
#     --model_card "openai/whisper-large-v2" \
#     --output_dir "/mnt/home/ntuspeechlabtaipei1/forbes/log_mel" \
#     --batch_size 64


# called by: raw_datasets["train"] = PreprocessedWhisperDataset(
#     features_dir="preprocessed_features",
#     audio_fpaths=audio_fpaths
# )


