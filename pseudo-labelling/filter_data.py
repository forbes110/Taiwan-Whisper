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

def resample_and_convert_to_flac(input_path):
    """
    Resample audio files (both .m4a and .flac) to 16kHz and convert to FLAC format.
    Delete the original file if conversion is successful.
    """
    try:
        start_time = time.time()
        
        # Determine new file path with .flac extension
        flac_path = input_path.replace(".m4a", ".flac")
        
        # Handle FLAC input separately to ensure resampling
        if input_path.endswith(".flac"):
            flac_path = input_path  # Keep the same path if already .flac

        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Resample to target sample rate if necessary
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            logging.info(f"Resampling {input_path} from {audio.frame_rate} to {TARGET_SAMPLE_RATE} Hz")
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # Export the audio to FLAC format with compression
        audio.export(
            flac_path,
            format="flac",
            parameters=["-compression_level", "8"]  # Optimize for speed and size
        )

        # Verify the new file and clean up the original
        if os.path.exists(flac_path) and os.path.getsize(flac_path) > 0:
            if input_path != flac_path:  # Avoid deleting if same file (FLAC)
                os.remove(input_path)  # Delete original file
            processing_time = time.time() - start_time
            return f"Converted and cleaned up: {flac_path} ({processing_time:.2f}s)"
        else:
            return f"Error: FLAC file not created properly for {input_path}"
        
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def process_chunk(file_chunk, max_workers):
    """Process a chunk of files using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resample_and_convert_to_flac, file): file for file in file_chunk}
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)

def process_directory(directory, max_workers=None):
    """
    Process audio files (both .m4a and .flac) in chunks using multiple processes and threads.
    """
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 2)

    # Find all .m4a and .flac files recursively
    audio_files = glob.glob(os.path.join(directory, '**', '*.m4a'), recursive=True) + \
                  glob.glob(os.path.join(directory, '**', '*.flac'), recursive=True)

    if not audio_files:
        logging.info("No audio files found.")
        return

    total_files = len(audio_files)
    logging.info(f"Found {total_files} audio files. Processing with {max_workers} workers...")

    # Process files in chunks
    chunks = [audio_files[i:i + CHUNK_SIZE] for i in range(0, total_files, CHUNK_SIZE)]

    with multiprocessing.Pool(processes=min(len(chunks), multiprocessing.cpu_count())) as pool:
        list(tqdm(
            pool.starmap(process_chunk, [(chunk, max_workers) for chunk in chunks]),
            total=len(chunks),
            desc="Processing chunks"
        ))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Resample and convert audio files to 16kHz FLAC.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker threads.")
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    try:
        process_directory(args.directory, args.max_workers)
        total_time = time.time() - start_time
        logging.info(f"Processing completed in {total_time:.2f} seconds")
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


# python3 filter_data.py --directory /mnt/dataset_1T/FTV_flac_selected --max_workers 4