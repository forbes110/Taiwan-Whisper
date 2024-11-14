import os
import glob
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging
import time
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_SAMPLE_RATE = 16000  # Target sample rate

def resample_audio(input_path, to_flac=True):
    """
    Resample audio files to 16kHz using ffmpeg. 
    If to_flac is True, convert to FLAC format.
    Handles output paths to avoid overwriting input files.
    Optionally replaces the original file after successful processing.
    """
    try:
        logging.debug(f"Processing {input_path}")
        start_time = time.time()
        base, ext = os.path.splitext(input_path)
        ext = ext.lower()

        # Determine the new file path based on the conversion option
        if to_flac:
            if ext == ".flac":
                # Input is already FLAC, resample and write to a temporary file
                output_path = base + "_resampled.flac"
            else:
                # Convert to FLAC
                output_path = base + ".flac"
        else:
            # Resample without converting to FLAC
            # Create a new file name to avoid overwriting
            output_path = base + "_resampled" + ext

        # Build the ffmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", input_path,
            "-ar", str(TARGET_SAMPLE_RATE),  # Set target sample rate
            "-ac", "1"  # Set number of audio channels to 1 (mono)
        ]

        if to_flac:
            ffmpeg_command += [
                "-c:a", "flac",
                "-compression_level", "8",
                output_path
            ]
        else:
            # Determine codec based on format
            if ext in ['.m4a', '.mp4']:
                codec = "aac"
            elif ext == '.wav':
                codec = "pcm_s16le"
            else:
                codec = "copy"  # Use stream copy for other formats
            ffmpeg_command += [
                "-c:a", codec,
                output_path
            ]

        # Execute the ffmpeg command
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if ffmpeg command was successful
        if result.returncode != 0:
            error_message = result.stderr.decode('utf-8')
            logging.error(f"ffmpeg error processing {input_path}: {error_message}")
            return f"Error processing {input_path}: {error_message}"

        # Verify the new file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Replace the original file with the new file if needed
            if to_flac and ext == ".flac":
                # If resampling a FLAC file, replace it with the resampled version
                shutil.move(output_path, input_path)
            elif to_flac and ext != ".flac":
                # If converting to FLAC from another format, delete the original
                os.remove(input_path)
            elif not to_flac:
                # If not converting to FLAC, replace the original file
                shutil.move(output_path, input_path)

            processing_time = time.time() - start_time
            return f"Processed: {input_path} ({processing_time:.2f}s)"
        else:
            logging.error(f"Output file not created properly for {input_path}")
            return f"Error: Output file not created properly for {input_path}"

    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def process_directory(input_path, to_flac=True, max_workers=None):
    """
    Process audio files (both .m4a and .flac) using multiple processes.
    The input_path can be a single file or a directory.
    """
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() + 4)

    # Determine if input_path is a file or directory
    if os.path.isfile(input_path):
        audio_files = [input_path]
    elif os.path.isdir(input_path):
        audio_files = glob.glob(os.path.join(input_path, '**', '*.m4a'), recursive=True) + \
                      glob.glob(os.path.join(input_path, '**', '*.flac'), recursive=True)
    else:
        logging.error(f"Invalid input path: {input_path}")
        return

    if not audio_files:
        logging.info("No audio files found.")
        return

    total_files = len(audio_files)
    logging.info(f"Found {total_files} audio files. Processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        # Submit all tasks
        futures = {executor.submit(resample_audio, file, to_flac): file for file in audio_files}
        
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing files", unit="file"):
            result = future.result()
            logging.info(result)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Resample and optionally convert audio files to 16kHz FLAC.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory containing audio files."
    )
    parser.add_argument(
        "--to_flac",
        action='store_true',
        help="Convert audio files to FLAC format. If not set, only resample the audio."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of worker processes."
    )
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    try:
        process_directory(args.input, to_flac=args.to_flac, max_workers=args.max_workers)
        total_time = time.time() - start_time
        logging.info(f"Processing completed in {total_time:.2f} seconds")
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
