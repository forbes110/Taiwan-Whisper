import os
import pandas as pd
import torch
import csv
import argparse
from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel  # Import the required modules

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperModel.")
    parser.add_argument("--dataset_path", type=str, default="/mnt/dataset_1T/tmp_dir/sample.tsv", help="Path to dataset.csv file.")
    parser.add_argument("--output_dir", type=str, default="/mnt/pseudo_label", help="Directory to save output CSV files.")
    parser.add_argument("--language", type=str, default="zh", help="Language code for transcription.")
    parser.add_argument("--log_progress", default=True, help="Display progress bars during transcription.")
    parser.add_argument("--model_size_or_path", type=str, default="tiny", help="Size or path of the Whisper model.")
    parser.add_argument("--compute_type", type=str, default="default", help="Compute type for CTranslate2 model.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing.")
    return parser.parse_args()

def load_dataset(dataset_path):
    """Load the dataset.csv and return a list of file paths."""
    df = pd.read_csv(dataset_path)
    return df["audio_path"].tolist()

def transcribe_audio_file(pipeline, audio_path, language="zh", log_progress=False):
    """Transcribe a single audio file and return the results."""
    segments, _ = pipeline.transcribe(
        audio_path,
        language=language,
        task="transcribe",  # Set task to transcribe (no translation)
        log_progress=log_progress
    )
    results = [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
    ]
    return results

def save_transcription_to_csv(transcriptions, output_csv):
    """Save the transcription results to a CSV file."""
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "text"])
        writer.writeheader()
        for item in transcriptions:
            writer.writerow(item)

def main():
    args = parse_args()
    print(args)

    """Main function to process all audio files listed in dataset.csv."""
    audio_files = load_dataset(args.dataset_path)

    # Initialize Whisper model and the pipeline
    model = WhisperModel(
        model_size_or_path=args.model_size_or_path,
        device="cuda" if torch.cuda.is_available() else "cpu",  # Auto-select device
        compute_type=args.compute_type,
        num_workers=args.num_workers  # Set number of workers for parallel processing
    )
    pipeline = BatchedInferencePipeline(model)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    for audio_path in audio_files:
        print(f"Processing: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        # Extract the file name without path and extension
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_csv = os.path.join(args.output_dir, f"{file_name}_transcription.csv")

        # Transcribe the audio file
        try:
            results = transcribe_audio_file(
                pipeline,
                audio_path,
                language=args.language,
                log_progress=args.log_progress
            )
            # Save the transcription results as a separate CSV file
            save_transcription_to_csv(results, output_csv)
            print(f"Transcription completed: {output_csv}")
        except Exception as e:
            print(f"Failed to transcribe {audio_path}, error: {e}")

if __name__ == "__main__":
    main()