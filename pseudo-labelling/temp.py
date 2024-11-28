import os
import argparse
import csv
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging
import gc

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('minnan_detection.log'),
        logging.StreamHandler()
    ]
)

mms_processor = None
mms_model = None
gpu_id = None

def cleanup_gpu():
    """Cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def init_worker(gpu_queue):
    """Initialize worker with GPU memory management"""
    global mms_processor, mms_model, gpu_id
    try:
        gpu_id = gpu_queue.get_nowait()
        cleanup_gpu()  # Clean before initialization
        
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'
        mms_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
        mms_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256").to(device)
        
        # Set model to eval mode to reduce memory usage
        mms_model.eval()
        
        # Use half precision to reduce memory usage
        if device != 'cpu':
            mms_model.half()
            
    except Exception as e:
        logging.error(f"Error initializing model on GPU {gpu_id}: {e}")
        cleanup_gpu()

def detect_minnan(audio_path):
    """Detect if the given audio file is in Taiwanese Hokkien (Min Nan)."""
    if not os.path.isfile(audio_path):
        logging.error(f"File '{audio_path}' does not exist.")
        return False, None

    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Move waveform to device
        waveform = waveform.to(device)

        # Process audio with mixed precision
        with torch.cuda.amp.autocast(enabled=device != 'cpu'):
            inputs = mms_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = mms_model(**inputs).logits

        lang_id = torch.argmax(outputs, dim=-1).item()
        detected_lang = mms_model.config.id2label[lang_id]

        # Cleanup
        del waveform, inputs, outputs
        cleanup_gpu()

        return detected_lang == "nan", detected_lang

    except Exception as e:
        logging.error(f"Error processing '{audio_path}': {e}")
        cleanup_gpu()
        return False, None

def process_file(audio_path, metadata_path):
    """Process single file with memory management"""
    try:
        is_minnan, lang = detect_minnan(audio_path)
        result = {
            'audio_path': audio_path,
            'detected_lang': lang,
            'is_minnan': is_minnan,
            'removed': False
        }
        
        if is_minnan:
            remove_path = os.path.sep.join(audio_path.split(os.path.sep)[-2:])
            
            # Read and update metadata
            with open(metadata_path, 'r') as f:
                lines = [line for line in f if remove_path not in line]
            
            with open(metadata_path, 'w') as f:
                f.writelines(lines)
                
            result['removed'] = True
            
        cleanup_gpu()
        return result

    except Exception as e:
        logging.error(f"Error in process_file for '{audio_path}': {e}")
        cleanup_gpu()
        return {
            'audio_path': audio_path,
            'detected_lang': None,
            'is_minnan': False,
            'removed': False
        }

def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Detect and optionally delete Taiwanese Hokkien (Min Nan) audio files.")
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--csv_output_dir", type=str, default="minnan_detected")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--metadata_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of files to process in each batch")
    args = parser.parse_args()

    # Collect all FLAC files
    flac_files = []
    for root, _, files in os.walk(args.directory):
        flac_files.extend([os.path.join(root, f) for f in files if f.lower().endswith('.flac')])

    total_files = len(flac_files)
    print(f"Found {total_files} FLAC files in '{args.directory}'.")
    
    channel_name = os.path.basename(args.directory)
    metadata_path = f"{args.metadata_dir}/{channel_name}.tsv"

    # Process files in batches
    detected_files = []
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    if not gpu_ids:
        gpu_ids = [None]

    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu in gpu_ids:
        gpu_queue.put(gpu)

    # Process in batches to manage memory
    for i in range(0, len(flac_files), args.batch_size):
        batch = flac_files[i:i + args.batch_size]
        
        with ProcessPoolExecutor(max_workers=args.num_workers, 
                               initializer=init_worker, 
                               initargs=(gpu_queue,)) as executor:
            
            future_to_file = {executor.submit(process_file, flac, metadata_path): flac for flac in batch}
            
            with tqdm(total=len(batch), desc=f"Batch {i//args.batch_size + 1}", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        if result['is_minnan']:
                            detected_files.append(result)
                    except Exception as e:
                        logging.error(f"Error in batch processing: {e}")
                    finally:
                        pbar.update(1)
                        cleanup_gpu()

        # Force cleanup between batches
        cleanup_gpu()

    # Save results
    os.makedirs(args.csv_output_dir, exist_ok=True)
    csv_file_path = os.path.join(args.csv_output_dir, "minnan_detected.csv")
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['audio_path', 'detected_lang', 'is_minnan', 'removed'])
        writer.writeheader()
        writer.writerows(detected_files)

    print(f"\nDetection completed. Found {len(detected_files)} 'minnan' speech files.")
    print(f"Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()