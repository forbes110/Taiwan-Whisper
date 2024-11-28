import os
import argparse
import csv
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing  # Import multiprocessing module
import logging

# Configure logging
logging.basicConfig(filename='minnan_detection_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Global variables to store model and processor references
mms_processor = None
mms_model = None
gpu_id = None

def init_worker(gpu_queue):
    """
    Initialize the model and processor in each worker process and assign to a GPU.
    Each worker pops a GPU ID from the gpu_queue.
    """
    global mms_processor
    global mms_model
    global gpu_id
    try:
        gpu_id = gpu_queue.get_nowait()
    except Exception as e:
        logging.error(f"Error assigning GPU to worker: {e}")
        gpu_id = None
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'
    try:
        mms_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
        mms_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256").to(device)
    except Exception as e:
        logging.error(f"Error initializing model on GPU {gpu_id}: {e}")

def detect_minnan(audio_path):
    """
    Detect if the given audio file is in Taiwanese Hokkien (Min Nan).

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        bool: True if the language is Taiwanese Hokkien, False otherwise.
        str: Detected language code.
    """
    if not os.path.isfile(audio_path):
        logging.error(f"File '{audio_path}' does not exist.")
        return False, None

    try:
        # Load the audio file and ensure it is 16kHz and mono
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to mono if it's stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure the model and input tensor are on the same device
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu'
        
        # Move waveform to the same device as the model
        waveform = waveform.to(device)

        # Process audio through the MMS model
        inputs = mms_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

        # Move inputs to the correct device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = mms_model(**inputs).logits
        lang_id = torch.argmax(outputs, dim=-1).item()
        detected_lang = mms_model.config.id2label[lang_id]

        # Check if the detected language is Taiwanese Hokkien
        if detected_lang == "nan":  # 'nan' is the ISO 639-3 code for Taiwanese Hokkien
            return True, "nan"
        else:
            return False, detected_lang

    except Exception as e:
        logging.error(f"Error processing '{audio_path}': {e}")
        return False, None

def process_file(audio_path, to_remove):
    """
    Process a single audio file: detect language and optionally delete files.

    Args:
        audio_path (str): Path to the audio file.
        to_remove (bool): Whether to delete the detected files.

    Returns:
        dict: Information about the processed file.
    """
    is_minnan, lang = detect_minnan(audio_path)
    result = {
        'audio_path': audio_path,
        'detected_lang': lang,
        'is_minnan': is_minnan
    }
    if is_minnan and to_remove:
        # Delete the audio file and its corresponding text file
        try:
            os.remove(audio_path)
            txt_path = os.path.splitext(audio_path)[0] + '.txt'
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            result['removed'] = True
        except Exception as e:
            logging.error(f"Error deleting '{audio_path}' and its corresponding files: {e}")
            result['removed'] = False
    else:
        result['removed'] = False
    return result

def main():
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Detect and optionally delete Taiwanese Hokkien (Min Nan) audio files.")
    parser.add_argument("--directory", type=str, help="Path to the directory containing audio files.", required=True)
    parser.add_argument("--to_remove", action="store_true", help="If set, delete detected audio files and their corresponding text files.")
    parser.add_argument("--csv_output_dir", type=str, default="minnan_detected", help="Path to the output CSV file.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes.")  # Default set to 8
    args = parser.parse_args()

    # Collect all FLAC files in the specified directory
    flac_files = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith('.flac'):
                flac_files.append(os.path.join(root, file))

    total_files = len(flac_files)
    print(f"Found {total_files} FLAC files in the directory '{args.directory}'.")

    detected_files = []

    # Get available GPU IDs
    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    if not gpu_ids:
        gpu_ids = [None]
    num_gpus = len(gpu_ids)

    if num_gpus < args.num_workers:
        print(f"Warning: Number of GPUs ({num_gpus}) is less than num_workers ({args.num_workers}). Some workers will not have a GPU assigned.")

    # Create a GPU assignment queue
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu in gpu_ids:
        gpu_queue.put(gpu)

    # Use ProcessPoolExecutor with initializer
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(gpu_queue,)) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, flac, args.to_remove): flac for flac in flac_files}
        
        # Initialize tqdm progress bar
        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                flac = future_to_file[future]
                try:
                    result = future.result()
                    if result['is_minnan']:
                        detected_files.append(result)
                except Exception as e:
                    logging.error(f"Error processing '{flac}': {e}")
                finally:
                    pbar.update(1)  # Update the progress bar

    # Ensure the output directory exists
    csv_output_dir = args.csv_output_dir
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    # Write detected files to CSV
    csv_file_path = os.path.join(csv_output_dir, "minnan_detected.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['audio_path', 'detected_lang', 'is_minnan', 'removed']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in detected_files:
            writer.writerow(entry)

    print(f"\nDetection completed. Found {len(detected_files)} 'minnan' speech files.")
    print(f"Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()

