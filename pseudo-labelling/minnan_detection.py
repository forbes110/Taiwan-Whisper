import os
import argparse
import csv
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import time
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('minnan_detection.log'),
        logging.StreamHandler()
    ]
)

class GPUMemoryManager:
    def __init__(self, threshold=0.8, device=None):
        self.threshold = threshold
        self.device = device
        self.peak_memory = 0
        self.cleanup_count = 0

    def get_current_memory(self):
        """Get current GPU memory usage"""
        if not torch.cuda.is_available() or self.device is None:
            return 0
        return torch.cuda.memory_allocated(self.device)

    def get_usage_ratio(self):
        """Get current GPU memory usage ratio."""
        if not torch.cuda.is_available() or self.device is None:
            return 0
        current_memory = torch.cuda.memory_reserved(self.device)
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        return current_memory / total_memory

    def check_and_cleanup(self, force=False):
        """Monitor and cleanup GPU memory if needed"""
        if not torch.cuda.is_available() or self.device is None:
            return
            
        usage_ratio = self.get_usage_ratio()
        if force or usage_ratio > self.threshold:
            self.force_cleanup()
            self.cleanup_count += 1
            logging.info(f"Performed cleanup #{self.cleanup_count}. Current memory usage ratio: {usage_ratio:.2f}")

    def force_cleanup(self):
        """Force GPU memory cleanup"""
        if not torch.cuda.is_available() or self.device is None:
            return
            
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        self.peak_memory = self.get_current_memory()
        
        usage_ratio = self.get_usage_ratio()
        if usage_ratio > self.threshold:
            logging.warning(f"Memory usage still high after cleanup: {usage_ratio:.2f}")

class MinNanDetector:
    def __init__(self, device=None):
        self.device = device
        self.memory_manager = GPUMemoryManager(threshold=0.6, device=device)
        self.processor = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model with memory optimization"""
        try:
            self.processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256")
            
            if self.device is not None:
                self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            self.memory_manager.force_cleanup()
            raise e

    def process_audio(self, audio_path):
        """Process a single audio file"""
        try:
            if torch.cuda.is_available() and self.device is not None:
                usage_ratio = self.memory_manager.get_usage_ratio()
                if usage_ratio > self.memory_manager.threshold:
                    self.memory_manager.check_and_cleanup()
                    tqdm.write(f"=== Clean up GPU for usage ratio {usage_ratio:.2f} > {self.memory_manager.threshold} ===")

            # Audio Loading with Exception Handling
            try:
                # Load and preprocess audio
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            except Exception as e:
                logging.error(f"Failed to load audio {audio_path}: {e}")
                return None, None  # Return None instead of raising

            
            if self.device is not None:
                waveform = waveform.to(self.device)
            
            # Process through model with mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
                    if self.device is not None:
                        inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    outputs = self.model(**inputs).logits
                    lang_id = torch.argmax(outputs, dim=-1).item()
            
            # Cleanup
            del waveform
            del inputs
            del outputs
            self.memory_manager.check_and_cleanup()
            
            detected_lang = self.model.config.id2label[lang_id]
            return detected_lang == "nan", detected_lang
            
        except Exception as e:
            logging.error(f"Error processing audio {audio_path}: {e}")
            self.memory_manager.force_cleanup()
            raise e

def update_metadata(metadata_path, files_to_remove): # MODIFIED: Changed parameter name
    """Update metadata file in a single process after all detection is complete"""
    try:
        # Create backup of metadata file
        backup_path = f"{metadata_path}.backup"
        if not os.path.exists(backup_path):
            with open(metadata_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logging.info(f"Created backup of metadata file at {backup_path}")

        # Read the entire metadata file
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
        
        original_count = len(lines)

        # Get paths to remove
        paths_to_remove = set()
        for file_path in files_to_remove: # MODIFIED: Changed iteration
            remove_path = os.path.sep.join(file_path.split(os.path.sep)[-2:])
            paths_to_remove.add(remove_path)

        # Filter out lines containing the paths
        filtered_lines = [line for line in lines if not any(path in line for path in paths_to_remove)]
        
        if len(filtered_lines) < len(lines):
            # Write back to file only if there are lines to remove
            with open(metadata_path, 'w') as f:
                f.writelines(filtered_lines)

            removed_count = original_count - len(filtered_lines)
            logging.info(f"Updated metadata file. Removed {removed_count} lines")
            print(f"Updated metadata file. Removed {removed_count} lines")
        else:
            logging.info("No lines needed to be removed from metadata file")
            print("No lines needed to be removed from metadata file")

    except Exception as e:
        logging.error(f"Error updating metadata: {e}")
        print(f"Error updating metadata: {e}")
        if os.path.exists(backup_path):
            with open(backup_path, 'r') as src, open(metadata_path, 'w') as dst:
                dst.write(src.read())
            logging.info("Restored metadata file from backup")
            print("Restored metadata file from backup")

# def update_metadata(metadata_path, detected_files):
#     """Update metadata file in a single process after all detection is complete"""
#     try:
#         # Create backup of metadata file
#         backup_path = f"{metadata_path}.backup"
#         if not os.path.exists(backup_path):
#             with open(metadata_path, 'r') as src, open(backup_path, 'w') as dst:
#                 dst.write(src.read())
#             logging.info(f"Created backup of metadata file at {backup_path}")

#         # Read the entire metadata file
#         with open(metadata_path, 'r') as f:
#             lines = f.readlines()
        
#         original_count = len(lines)

#         # Get paths to remove
#         paths_to_remove = set()
#         for result in detected_files:
#             if result['is_minnan']:
#                 remove_path = os.path.sep.join(result['audio_path'].split(os.path.sep)[-2:])
#                 paths_to_remove.add(remove_path)

#         # Filter out lines containing the paths
#         filtered_lines = [line for line in lines if not any(path in line for path in paths_to_remove)]
        
#         if len(filtered_lines) < len(lines):
#             # Write back to file only if there are lines to remove
#             with open(metadata_path, 'w') as f:
#                 f.writelines(filtered_lines)

#             # Update removed status in results
#             for result in detected_files:
#                 if result['is_minnan']:
#                     result['removed'] = True

#             removed_count = original_count - len(filtered_lines)
#             logging.info(f"Updated metadata file. Removed {removed_count} lines")
#             print(f"Updated metadata file. Removed {removed_count} lines")
#         else:
#             logging.info("No lines needed to be removed from metadata file")
#             print("No lines needed to be removed from metadata file")

#     except Exception as e:
#         logging.error(f"Error updating metadata: {e}")
#         print(f"Error updating metadata: {e}")
#         # Restore from backup if available
#         if os.path.exists(backup_path):
#             with open(backup_path, 'r') as src, open(metadata_path, 'w') as dst:
#                 dst.write(src.read())
#             logging.info("Restored metadata file from backup")
#             print("Restored metadata file from backup")

# Global variables for multiprocessing
detector = None
gpu_id = None

def init_worker(gpu_queue):
    """Initialize worker process"""
    global detector, gpu_id
    try:
        gpu_id = gpu_queue.get_nowait()
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'
        detector = MinNanDetector(device)
        logging.info(f"Initialized worker with GPU {gpu_id}")
    except Exception as e:
        logging.error(f"Error initializing worker: {e}")
        gpu_id = None
        device = 'cpu'
        detector = MinNanDetector(device)

def process_file(audio_path):
    """Process a single file with retry mechanism"""
    global detector
    max_retries = 2
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(retry_delay * attempt)
                detector.memory_manager.force_cleanup()

            try: # MODIFIED: Added try-except for audio loading
                is_minnan, lang = detector.process_audio(audio_path)
                if lang is None:  # Audio loading failed
                    return True # MODIFIED: Return True to indicate removal from metadata only
                return False if not is_minnan else {'audio_path': audio_path, 'detected_lang': lang, 'is_minnan': True, 'removed': False}
            except Exception as e:
                logging.error(f"Audio loading error for {audio_path}: {e}")
                return True # MODIFIED: Return True for failed audio loading

        except Exception as e:
            detector.memory_manager.force_cleanup()
            if attempt == max_retries - 1:
                return True # MODIFIED: Return True for failed attempts


# def process_file(audio_path):
#     """Process a single file with retry mechanism"""
#     global detector
#     max_retries = 2
#     retry_delay = 2
    
#     for attempt in range(max_retries):
#         try:
#             if attempt > 0:
#                 tqdm.write(f"\n=== Retrying {audio_path} (attempt {attempt + 1}/{max_retries}) ===")
#                 time.sleep(retry_delay * attempt)
#                 detector.memory_manager.force_cleanup()

#             is_minnan, lang = detector.process_audio(audio_path)
#             result = {
#                 'audio_path': audio_path,
#                 'detected_lang': lang,
#                 'is_minnan': is_minnan,
#                 'removed': False
#             }
#             return result

#         except Exception as e:
#             detector.memory_manager.force_cleanup()
#             if attempt < max_retries - 1:
#                 logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
#                 continue
#             else:
#                 logging.error(f"All {max_retries} attempts failed for {audio_path}: {e}")
#                 return {
#                     'audio_path': audio_path,
#                     'detected_lang': None,
#                     'is_minnan': False,
#                     'removed': False
#                 }

# def main():
#     parser = argparse.ArgumentParser(description="Detect and optionally delete Taiwanese Hokkien (Min Nan) audio files.")
#     parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing audio files.")
#     parser.add_argument("--csv_output_dir", type=str, default="minnan_detected", help="Path to the output CSV file.")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes.")
#     parser.add_argument("--metadata_dir", type=str, default=".", help="Metadata Directory")
#     args = parser.parse_args()

#     # Set up multiprocessing
#     multiprocessing.set_start_method('spawn', force=True)

#     # Collect audio files
#     flac_files = []
#     for root, _, files in os.walk(args.directory):
#         for file in files:
#             if file.lower().endswith('.flac'):
#                 flac_files.append(os.path.join(root, file))

#     total_files = len(flac_files)
#     print(f"Found {total_files} FLAC files in '{args.directory}'")

#     channel_name = os.path.basename(args.directory)
#     metadata_path = f"{args.metadata_dir}/{channel_name}.tsv"

#     # Set up GPU queue
#     gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
#     if not gpu_ids:
#         gpu_ids = [None]
    
#     manager = multiprocessing.Manager()
#     gpu_queue = manager.Queue()
#     for gpu in gpu_ids * (args.num_workers // len(gpu_ids) + 1):  # Ensure enough GPU assignments
#         gpu_queue.put(gpu)

#     detected_files = []
#     failed_files = []

#     # Process files
#     with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(gpu_queue,)) as executor:
#         future_to_file = {executor.submit(process_file, flac): flac for flac in flac_files}
        
#         with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
#             for future in as_completed(future_to_file):
#                 flac = future_to_file[future]
#                 try:
#                     result = future.result()
#                     if result['detected_lang'] is None:
#                         failed_files.append(flac)
#                     elif result['is_minnan']:
#                         detected_files.append(result)
#                 except Exception as e:
#                     failed_files.append(flac)
#                     logging.error(f"Error processing '{flac}': {e}")
#                 finally:
#                     pbar.update(1)

#     # Update metadata after all processing is complete
#     if detected_files:
#         update_metadata(metadata_path, detected_files)

#     # Save results
#     os.makedirs(args.csv_output_dir, exist_ok=True)
    
#     if failed_files:
#         failed_list_path = os.path.join(args.csv_output_dir, "failed_files.txt")
#         with open(failed_list_path, 'w') as f:
#             for file in failed_files:
#                 f.write(f"{file}\n")
#         print(f"\nFailed files saved to: {failed_list_path}")

#     csv_file_path = os.path.join(args.csv_output_dir, "minnan_detected.csv")
#     with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
#         fieldnames = ['audio_path', 'detected_lang', 'is_minnan', 'removed']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#         for entry in detected_files:
#             writer.writerow(entry)

#     print(f"\nDetection completed. Found {len(detected_files)} 'minnan' speech files.")
#     print(f"Results saved to '{csv_file_path}'.")

# if __name__ == "__main__":
#     main()


def main():
    parser = argparse.ArgumentParser(description="Detect and optionally delete Taiwanese Hokkien (Min Nan) audio files.")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing audio files.")
    parser.add_argument("--csv_output_dir", type=str, default="minnan_detected", help="Path to the output CSV file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--metadata_dir", type=str, default=".", help="Metadata Directory")
    args = parser.parse_args()

    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Collect audio files
    flac_files = []
    for root, _, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith('.flac'):
                flac_files.append(os.path.join(root, file))

    total_files = len(flac_files)
    print(f"Found {total_files} FLAC files in '{args.directory}'")

    channel_name = os.path.basename(args.directory)
    metadata_path = f"{args.metadata_dir}/{channel_name}.tsv"

    # Set up GPU queue
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    if not gpu_ids:
        gpu_ids = [None]
    
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu in gpu_ids * (args.num_workers // len(gpu_ids) + 1):  # Ensure enough GPU assignments
        gpu_queue.put(gpu)

    detected_files = []
    # failed_files = []
    files_to_remove = []

    # Process files
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(gpu_queue,)) as executor:
        future_to_file = {executor.submit(process_file, flac): flac for flac in flac_files}
        
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                flac = future_to_file[future]
                try:
                    result = future.result()
                    # if result['detected_lang'] is None:
                    #     failed_files.append(flac)
                    # elif result['is_minnan']:
                    #     detected_files.append(result)
                    if isinstance(result, bool) and result: # MODIFIED: Check for metadata removal only
                        files_to_remove.append(flac)
                    elif isinstance(result, dict): # MODIFIED: Check for Minnan detection
                        detected_files.append(result)
                except Exception as e:
                    # failed_files.append(flac)
                    files_to_remove.append(flac) # MODIFIED: Add failed files to removal
                    logging.error(f"Error processing '{flac}': {e}")
                finally:
                    pbar.update(1)
    files_to_update = files_to_remove + [result['audio_path'] for result in detected_files] # MODIFIED
    if files_to_update:
        update_metadata(metadata_path, files_to_update)

    
    if detected_files: # MODIFIED: Only save Minnan files
        os.makedirs(args.csv_output_dir, exist_ok=True)
        csv_file_path = os.path.join(args.csv_output_dir, "minnan_detected.csv")
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['audio_path', 'detected_lang', 'is_minnan', 'removed']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for entry in detected_files:
                writer.writerow(entry)
        print(f"\nDetection completed. Found {len(detected_files)} 'minnan' speech files.")
        print(f"Results saved to '{csv_file_path}'.")

        if files_to_remove: # MODIFIED: Print removal stats
            print(f"\nRemoved {len(files_to_remove)} invalid/failed files from metadata")

if __name__ == "__main__":
    main()