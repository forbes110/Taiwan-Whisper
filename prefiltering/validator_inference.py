import os
import numpy as np
import os.path as osp
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import soundfile as sf
from transformers import (
    AddedToken,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

sampling_rate = 16000
logger = get_logger(__name__)

class WhisperSmallModelDetector:
    def __init__(self, small_model_card='openai/whisper-base', accelerator=None):

        self.model = WhisperForConditionalGeneration.from_pretrained(small_model_card, low_cpu_mem_usage=True)
        self.processor = WhisperProcessor.from_pretrained(small_model_card)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(small_model_card, use_fast=True)
        self.accelerator = accelerator

        timestamps = [AddedToken(f"<|{i * 0.02:.2f}|>", lstrip=False, rstrip=False) for i in range(1500 + 1)]
        self.tokenizer.add_tokens(timestamps)
        self.tokenizer.set_prefix_tokens(language='zh', task='transcribe', predict_timestamps=True)

        self.gen_kwargs = {
            "max_length": 448,
            "num_beams": 1,
            "return_timestamps": True,
            "language": 'zh',
            "task": 'transcribe',
        }
        self.input_padding = "longest"
        self.model.eval()
        if self.accelerator is not None:
            self.model = self.model.to(self.accelerator.device)

    def collate_fn(self, features):
        # 提取音頻特徵並準備批次
        inputs = self.processor.feature_extractor(
            [feature['array'] for feature in features],
            sampling_rate=sampling_rate,
        )
        input_features = {'input_features': inputs.input_features}
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )
        batch['idx'] = torch.tensor([feature['idx'] for feature in features], dtype=torch.long).to(batch['input_features'].device)
        return batch

    def generate(self, batch):
        # 確保 input_features 被移動到與模型相同的設備
        input_features = batch["input_features"].to(self.accelerator.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_features, **self.gen_kwargs)
        preds_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, decode_with_timestamps=False)
        return batch['idx'], preds_str

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

class PreFilterASRDataset(Dataset):
    def __init__(self, metadata_fpath, root=None):
        self.metadata_fpath = metadata_fpath
        logger.info(f"Loading dataset from {metadata_fpath}")
        self.audio_fpaths = load_audio_fpaths(metadata_fpath, root=root)

    def __getitem__(self, idx):

        audio_fpath = self.audio_fpaths[idx]
        logger.info(f"Loading audio from {audio_fpath}")
        audio_data, sr = sf.read(audio_fpath)

        assert sr == sampling_rate, f"Sampling rate {sr} is not {sampling_rate}Hz"
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if len(audio_data) != 30 * sampling_rate:
            if len(audio_data) < 30 * sampling_rate:
                audio_data = np.pad(audio_data, (0, 30 * sampling_rate - len(audio_data)))
            else:
                audio_data = audio_data[:30 * sampling_rate]
        feature = {'idx': idx, 'path': audio_fpath, 'array': audio_data}
        return feature

    def __len__(self):
        return len(self.audio_fpaths)

def main(args):
    print(args)

    accelerator = Accelerator(cpu=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = PreFilterASRDataset(args.manifest, root=args.root)
    if accelerator.is_main_process:
        print(f"Total samples: {len(dataset)}")

    hallucination_detector = WhisperSmallModelDetector(small_model_card=args.validator, accelerator=accelerator)

    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False  # 如果需要打亂，設置為 True
    )

    dataloader_for_test = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,  
        num_workers=8,
        collate_fn=hallucination_detector.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    if accelerator.is_main_process:
        print(f"Dataloader size: {len(dataloader_for_test)}")

    if accelerator.is_main_process:
        steps_inference_progress_bar = tqdm(
            total=len(dataloader_for_test),
            desc="Inference segments ... ",
            position=0,
            disable=not accelerator.is_local_main_process
        )

    accelerator.wait_for_everyone()

    # 每個進程寫入自己的文件
    process_rank = accelerator.process_index
    output_file_path = f"{args.output_dir}/validator_inference_rank{process_rank}.txt"
    fw = open(output_file_path, "w")

    for batch in dataloader_for_test:
        idxs, preds_str = hallucination_detector.generate(batch)

        for idx, pred_str in zip(idxs.cpu().tolist(), preds_str):
            fw.write(f"{idx}\t{pred_str}\n")
        fw.flush()

        if accelerator.is_main_process:
            steps_inference_progress_bar.update(1)

    fw.close()

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        merge_and_sort_results(args.output_dir, accelerator.num_processes)
        steps_inference_progress_bar.close()

    print("Everything is done!")
    
def merge_and_sort_results(output_dir, num_processes):
    """
    Merge results from all processes, sort by ID, and remove duplicates.
    """
    # Dictionary to store unique entries (id -> prediction)
    all_results = {}
    
    # Read all rank files and store in dictionary
    for rank in range(num_processes):
        rank_file = f"{output_dir}/validator_inference_rank{rank}.txt"
        with open(rank_file, "r") as rf:
            for line in rf:
                idx, pred = line.strip().split('\t', 1)
                idx = int(idx)
                
                # Store the prediction (last one wins in case of duplicates)
                all_results[idx] = pred
                
    # Write sorted results to final file
    with open(f"{output_dir}/validator_inference.txt", "w") as final_fw:
        for idx in sorted(all_results.keys()):
            final_fw.write(f"{idx}\t{all_results[idx]}\n")
            
    # Clean up rank files
    for rank in range(num_processes):
        rank_file = f"{output_dir}/validator_inference_rank{rank}.txt"
        os.remove(rank_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Model Validator Inference Script")

    parser.add_argument(
        "--validator",
        default="openai/whisper-tiny",
        help="Model card for the Whisper validator (e.g., 'openai/whisper-base')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory for audio files. If not provided, the first line of the manifest will be used.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        required=True,
        help="Path to the manifest file containing audio file paths",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        help="Directory to save the output results",
    )
    args = parser.parse_args()

    main(args)
