import os
import pandas as pd
import torch
import csv
import argparse
from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel  # Import the required modules
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
import tokenizers
import concurrent.futures

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperModel.")
    parser.add_argument("--dataset_path", type=str, default="/mnt/dataset_1T/tmp_dir/sample.tsv", help="Path to dataset.csv file.")
    parser.add_argument("--batched_mode", action="store_true", help="Whether to use batched inference.")
    parser.add_argument("--output_dir", type=str, default="/mnt/pseudo_label", help="Directory to save output CSV files.")
    parser.add_argument("--language", type=str, default="zh", help="Language code for transcription.")
    parser.add_argument("--log_progress", default=True, help="Display progress bars during transcription.")
    parser.add_argument("--model_size", type=str, default="tiny", help="Size or path of the Whisper model.")
    parser.add_argument("--compute_type", type=str, default="default", help="Compute type for CTranslate2 model.")
    parser.add_argument('--chunk_length', type=int, default=5, help='The length of audio segments. If it is not None, it will overwrite the default chunk_length of the FeatureExtractor.')
    parser.add_argument('--batch_size', type=int, default=64, help='The maximum number of parallel requests to model for decoding.')
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing.")
    
    return parser.parse_args()

def load_dataset(dataset_path):
    """Load the dataset.csv and return a list of file paths."""
    df = pd.read_csv(dataset_path, sep='\t')  # Assuming TSV as per the default path
    return df["audio_path"].tolist()

def transcribe_audio_file(pipeline, audio_path, language="zh", log_progress=False, batch_size=64):
    """Transcribe a single audio file and return the results."""
    
    segments, _ = pipeline.transcribe(
        audio=audio_path,
        task="transcribe",  # Set task to transcribe (no translation)
        log_progress=log_progress,
        batch_size=batch_size,
    )
    results = [
        {"start": f"{segment.start}", "end": f"{segment.end}", "text": segment.text}
        for segment in segments
    ]
    return results


def transcribe_audio_file_sequential(model, audio_path, language="zh", log_progress=False, chunk_length=5):
    """Transcribe a single audio file using sequential inference and return the results."""
    segments, _ = model.transcribe(
        audio_path, 
        task="transcribe", 
        language=language,
        multilingual=True,
        output_language="hybrid",
        beam_size=5,
        chunk_length=chunk_length,
    )
    results = [
        {"start": f"{segment.start}", "end": f"{segment.end}", "text": segment.text}
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

def worker_transcribe(audio_path, output_dir, model_size, compute_type, language, log_progress, chunk_length):
    """Worker function to transcribe a single audio file."""
    try:
        # Initialize the model inside the worker to avoid issues with multiprocessing
        model = WhisperModel(
            model_size_or_path=model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            device_index=[0],
            compute_type=compute_type,
            num_workers=1  # Each worker handles one task at a time
        )
        
        results = transcribe_audio_file_sequential(model, audio_path, language, log_progress, chunk_length=5)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the file name without path and extension
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_csv = os.path.join(output_dir, f"{file_name}.csv")
        
        # Save the transcription results as a separate CSV file
        save_transcription_to_csv(results, output_csv)
        print(f"Transcription completed: {output_csv}")
    except Exception as e:
        print(f"Failed to transcribe {audio_path}, error: {e}")

def main():
    args = parse_args()
    print(args)
    model_size = args.model_size

    """Main function to process all audio files listed in dataset.csv."""
    audio_files = load_dataset(args.dataset_path)
                   
    # Initialize Whisper model and the pipeline
    if args.batched_mode:
        model = WhisperModel(
            model_size_or_path=model_size,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            device_index = [0],
            compute_type=args.compute_type,
            num_workers=args.num_workers  # Set number of workers for parallel processing
        )
        
        print(f"Using device: {model.device}", flush=True)
        
        if args.batched_mode:
            print("Using batched inference", flush=True)
                
            tokenizer = Tokenizer(
                multilingual=True,
                task="transcribe",
                language="zh",
                tokenizer=tokenizers.Tokenizer.from_pretrained(f"openai/whisper-{model_size}")
            )
            
            pipeline = BatchedInferencePipeline(
                model, 
                use_vad_model=True,  # Enable VAD model
                chunk_length=args.chunk_length,   
                language=args.language,
                tokenizer=tokenizer
            )

            # Ensure the output directory exists
            os.makedirs(args.output_dir, exist_ok=True)

            for audio_path in audio_files:
                print(f"Processing: {audio_path}")
                if not os.path.exists(audio_path):
                    print(f"File not found: {audio_path}")
                    continue

                # Extract the file name without path and extension
                file_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_csv = os.path.join(args.output_dir, f"{file_name}.csv")

                # Transcribe the audio file
                try:
                    results = transcribe_audio_file(
                        pipeline,
                        audio_path=audio_path,
                        language=args.language,
                        log_progress=args.log_progress,
                        batch_size=args.batch_size,
                        
                    )
                    # Save the transcription results as a separate CSV file
                    save_transcription_to_csv(results, output_csv)
                    print(f"Transcription completed: {output_csv}")
                except Exception as e:
                    print(f"Failed to transcribe {audio_path}, error: {e}")
    else:
        print("Using sequential inference", flush=True)
        
        # parallel over video level
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # submit all tasks to executor
            futures = [
                executor.submit(
                    worker_transcribe,
                    audio_path=audio_path,
                    output_dir=args.output_dir,
                    model_size=args.model_size,
                    compute_type=args.compute_type,
                    language=args.language,
                    log_progress=args.log_progress,
                    chunk_length=args.chunk_length
                )
                for audio_path in audio_files
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()