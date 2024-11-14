import os
import argparse
import csv
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar

# Global variables to store model and processor references
mms_processor = None
mms_model = None

def init_model():
    """
    Load the model and processor in each worker process and assign to global variables.
    """
    global mms_processor
    global mms_model
    mms_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
    mms_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256").to('cuda')

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
        print(f"Error: File '{audio_path}' does not exist.")
        return False, None

    try:
        # Load the audio file and ensure it is 16kHz and mono
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to mono if it's stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure that both model and input tensor are on the same device
        device = next(mms_model.parameters()).device  # Get model device (GPU in this case)
        
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

        # Check if detected language is Taiwanese Hokkien
        if detected_lang == "nan":  # 'nan' is the ISO 639-3 code for Taiwanese Hokkien
            return True, "nan"
        else:
            return False, detected_lang

    except Exception as e:
        print(f"Error processing '{audio_path}': {e}")
        return False, None

def process_file(audio_path, to_remove):
    """
    Process a single audio file: detect language and optionally delete files.

    Args:
        audio_path (str): Path to the audio file.
        to_remove (bool): Whether to delete detected files.

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
        # Delete audio file and corresponding text file
        try:
            os.remove(audio_path)
            txt_path = os.path.splitext(audio_path)[0] + '.txt'
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            result['removed'] = True
        except Exception as e:
            print(f"Error deleting '{audio_path}' and its corresponding files: {e}")
            result['removed'] = False
    else:
        result['removed'] = False
    return result

def main():
    """
    Main function to handle user input and display results.
    """
    parser = argparse.ArgumentParser(description="Detect and optionally delete Taiwanese Hokkien (Min Nan) audio files.")
    parser.add_argument("--directory", type=str, help="Path to the directory containing audio files.", required=True)
    parser.add_argument("--to_remove", action="store_true", help="If set, delete detected audio files and their corresponding text files.")
    parser.add_argument("--csv_output_dir", type=str, default="minnan_detected", help="Path to the output CSV file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes.")
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

    # Use ProcessPoolExecutor for parallel processing with a progress bar
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_model) as executor:
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
                    print(f"Error processing '{flac}': {e}")
                finally:
                    pbar.update(1)  # Update the progress bar for each completed task

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
