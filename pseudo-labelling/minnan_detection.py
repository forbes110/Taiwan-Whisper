# import os
# from pydub import AudioSegment
# from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
# import torch
# import torchaudio
# import io

# def detect_minnan(audio_path):
#     if not os.path.isfile(audio_path):
#         print(f"Error: File '{audio_path}' does not exist.")
#         return False, None

#     try:
#         # Load the MMS model for language identification
#         mms_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
#         mms_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256")

#         # Load and sample the audio
#         audio = AudioSegment.from_file(audio_path)
#         sampled_audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz and mono

#         # Convert sampled audio to bytes-like object for in-memory processing
#         audio_buffer = io.BytesIO()
#         sampled_audio.export(audio_buffer, format="wav")
#         audio_buffer.seek(0)  # Reset buffer position to the start

#         # Load the audio waveform directly from the in-memory buffer
#         waveform, sample_rate = torchaudio.load(audio_buffer)

#         # Resample if necessary
#         if sample_rate != mms_processor.sampling_rate:
#             waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=mms_processor.sampling_rate)(waveform)

#         # Process audio through MMS
#         inputs = mms_processor(waveform.squeeze(), sampling_rate=mms_processor.sampling_rate, return_tensors="pt")
#         with torch.no_grad():
#             outputs = mms_model(**inputs).logits
#         lang_id = torch.argmax(outputs, dim=-1)[0].item()
#         detected_lang = mms_model.config.id2label[lang_id]

#         # Check if detected language is Taiwanese Hokkien
#         if detected_lang == "nan":  # 'nan' is the ISO 639-3 code for Taiwanese Hokkien
#             return True, "nan"
        
#         else:
#             return False, detected_lang

#     except Exception as e:
#         print(f"An error occurred while processing '{audio_path}': {e}")
#         return False, None

# def main():
#     """
#     Main function to handle user input and display results.
#     """
#     audio_file_1 = "/content/__9N8mzmiZ0_0-473264.flac" # no
#     audio_file_2 = "/content/__ob4PLXPnw_51449808-51900688.flac" # yes
#     audio_file_3 = "/content/__ob4PLXPnw_54217840-54662784.flac" # yes
#     audio_file_4 = "/content/__ob4PLXPnw_54662784-55108848.flac" # yes

#     for  audio_file in [audio_file_1, audio_file_2, audio_file_3, audio_file_4]:
#         is_taiwanese, language_code = detect_minnan(audio_file)
        
#         print("------------------------------------------------------------------------------------------------------")
#         if is_taiwanese:
#             print(f"The audio file '{audio_file}' is detected as Taiwanese Hokkien (Language Code: {language_code}).")
#         elif language_code:
#             print(f"The audio file '{audio_file}' is NOT Taiwanese Hokkien. Detected Language Code: {language_code}.")
#         else:
#             print(f"Could not determine the language of the audio file '{audio_file}'.")
#         print("------------------------------------------------------------------------------------------------------")
        

# if __name__ == "__main__":
#     main()
import os
import argparse
import csv
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global variables to store model references
mms_processor = None
mms_model = None

def init_model(processor, model):
    """
    Assign the pre-loaded model and processor to global variables for each worker process.
    """
    global mms_processor
    global mms_model
    mms_processor = processor
    mms_model = model

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

        # Process audio through MMS model
        inputs = mms_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
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
    parser.add_argument("--directory", type=str, help="Path to the directory containing audio files.")
    parser.add_argument("--to_remove", action="store_true", help="If set, delete detected audio files and their corresponding text files.")
    parser.add_argument("--csv_output_dir", type=str, default="minnan_detected", help="Path to the output CSV file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes.")
    args = parser.parse_args()

    # Load the model and processor once in the main process
    processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256")

    # Collect all FLAC files in the specified directory
    flac_files = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith('.flac'):
                flac_files.append(os.path.join(root, file))

    print(f"Found {len(flac_files)} FLAC files in the directory '{args.directory}'.")

    detected_files = []

    # Use ProcessPoolExecutor for parallel processing, passing the pre-loaded model and processor
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_model, initargs=(processor, model)) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, flac, args.to_remove): flac for flac in flac_files}
        for future in as_completed(future_to_file):
            flac = future_to_file[future]
            try:
                result = future.result()
                if result['is_minnan']:
                    detected_files.append(result)
                print(f"Processed: {flac}")
            except Exception as e:
                print(f"Error processing '{flac}': {e}")

    csv_output_dir = args.csv_output_dir
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    # Write detected files to CSV
    with open(f"{csv_output_dir}/minnan_detected.csv", mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['audio_path', 'detected_lang', 'is_minnan', 'removed']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in detected_files:
            writer.writerow(entry)

    print(f"Detection completed. Found {len(detected_files)} 'minnan' speech files.")
    print(f"Results saved to '{args.csv_output}'.")

if __name__ == "__main__":
    main()
