import os
import glob
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_SAMPLE_RATE = 16000  # Target sample rate
CHUNK_SIZE = 100  # Process files in chunks for better memory management

def resample_audio(input_path, to_flac=True):
    """
    Resample audio files to 16kHz. If to_flac is True, convert to FLAC format.
    Delete the original file if conversion is successful and to_flac is True.
    """
    try:
        start_time = time.time()
        base, ext = os.path.splitext(input_path)
        ext = ext.lower()

        # Determine the new file path based on the conversion option
        if to_flac:
            if ext == ".flac":
                output_path = input_path  # Already FLAC
            else:
                output_path = base + ".flac"
        else:
            output_path = input_path  # Overwrite the original file
            format = ext.lstrip('.')  # Extract format from extension
            if format == 'm4a':
                format = 'mp4'  # pydub uses 'mp4' for .m4a files

        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)

        # Resample if necessary
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            logging.info(f"Resampling {input_path} from {audio.frame_rate} to {TARGET_SAMPLE_RATE} Hz")
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # Export the audio
        if to_flac:
            audio.export(output_path, format="flac", parameters=["-compression_level", "8"])
        else:
            audio.export(output_path, format=format)

        # Verify the new file and clean up the original
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if to_flac and input_path != output_path:
                os.remove(input_path)  # Delete original file
            processing_time = time.time() - start_time
            return f"Processed: {output_path} ({processing_time:.2f}s)"
        else:
            return f"Error: Output file not created properly for {input_path}"

    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def process_chunk(file_chunk, max_workers, to_flac, progress_bar):
    """Process a chunk of files with a progress bar."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resample_audio, file, to_flac): file for file in file_chunk}
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)
            progress_bar.update(1)  # Update progress bar

def process_directory(input_path, to_flac=True, max_workers=None):
    """
    Process audio files (both .m4a and .flac) in chunks using multiple processes and threads.
    The input_path can be a single file or a directory.
    """
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 2)

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

    chunks = [audio_files[i:i + CHUNK_SIZE] for i in range(0, total_files, CHUNK_SIZE)]

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        with multiprocessing.Pool(processes=min(len(chunks), multiprocessing.cpu_count())) as pool:
            for chunk in chunks:
                pool.apply_async(process_chunk, (chunk, max_workers, to_flac, pbar))

            pool.close()
            pool.join()

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
        help="Number of worker threads per process."
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
