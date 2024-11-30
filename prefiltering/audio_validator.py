import soundfile as sf
import numpy as np
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from collections import Counter
import shutil
import tempfile

def read_audio(audio_path):
    """
    Attempts to read an audio file.
    Returns (None, duration) if successful, or (error_message, None) if an exception occurs.
    """
    try:
        audio_data, file_sr = sf.read(audio_path, dtype='float32')
        duration = len(audio_data) / file_sr  # Calculate duration in seconds
        return None, duration  # No error, return duration
    except Exception as e:
        print("------------------------------------------------------------------------------------------------------------------")
        return f"{audio_path}\t{type(e).__name__}: {str(e)}", None

def process_tsv(input_path, num_workers):
    """
    Reads the TSV file and returns a list of absolute audio file paths.
    Assumes the first line is the prefix path and subsequent lines are relative paths.
    """
    audio_paths = []
    prefix = ""

    with open(input_path, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row_number, row in enumerate(reader, start=1):
            if not row:
                print(f"Row {row_number} is empty. Skipping.")
                continue

            if row_number == 1:
                prefix = row[0].strip()
                if not prefix:
                    print(f"Row {row_number} has an empty prefix path. Exiting.")
                    return audio_paths, prefix
                if not os.path.isabs(prefix):
                    prefix = os.path.abspath(prefix)
                print(f"Prefix path: {prefix}")
                continue

            relative_path = row[0].strip()
            if relative_path:
                absolute_path = os.path.join(prefix, relative_path)
                audio_paths.append((absolute_path, relative_path))
            else:
                print(f"Row {row_number} has an empty path. Skipping.")

    return audio_paths, prefix

def update_tsv_file(input_path, failed_paths, prefix):
    """
    Creates a new TSV file excluding the failed paths.
    Returns the path to the new TSV file.
    """
    failed_relative_paths = {os.path.relpath(path, prefix) for path in failed_paths}
    temp_fd, temp_path = tempfile.mkstemp(suffix='.tsv')
    os.close(temp_fd)

    with open(input_path, 'r', newline='') as original, open(temp_path, 'w', newline='') as temp_file:
        reader = csv.reader(original, delimiter='\t')
        writer = csv.writer(temp_file, delimiter='\t')
        
        # Write the prefix (first line)
        first_row = next(reader)
        writer.writerow(first_row)
        
        # Write the remaining paths, excluding failed ones
        for row in reader:
            if row and row[0].strip() and row[0].strip() not in failed_relative_paths:
                writer.writerow(row)

    # Replace the original file with the new one
    shutil.move(temp_path, input_path)
    print(f"\nUpdated TSV file: {input_path}")
    print(f"Removed {len(failed_paths)} failed paths")

def format_duration(seconds):
    """
    Formats duration in seconds to HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def write_duration_report(output_path, total_duration, durations_by_file):
    """
    Writes duration information to a file
    """
    with open(output_path, 'w') as f:
        f.write("=== Audio Duration Report ===\n\n")
        f.write(f"Total Duration: {format_duration(total_duration)} (HH:MM:SS)\n")
        f.write(f"Total Seconds: {total_duration:.2f}\n\n")
        f.write("=== Individual File Durations ===\n\n")
        for file_path, duration in durations_by_file:
            f.write(f"{file_path}\t{format_duration(duration)} ({duration:.2f}s)\n")

def main():
    parser = argparse.ArgumentParser(description="Check .flac audio files for errors.")
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the input TSV file containing audio file paths.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to the output error log file.'
    )
    parser.add_argument(
        '--duration_output',
        type=str,
        help='Path to output file for duration information.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers to use for processing. Recommended: Number of CPU cores or slightly higher.'
    )
    parser.add_argument(
        '--calculate_duration',
        action='store_true',
        help='Calculate and display total duration of all audio files'
    )
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_path):
        print(f"Input TSV file does not exist: {args.input_path}")
        return

    print(f"Processing TSV file: {args.input_path}")
    print(f"Using {args.num_workers} worker(s) for parallel processing.")
    print(f"Errors will be logged to: {args.output_path}\n")

    audio_paths, prefix = process_tsv(args.input_path, args.num_workers)
    total_files = len(audio_paths)
    print(f"Total audio files to process: {total_files}\n")

    if total_files == 0:
        print("No audio files to process. Exiting.")
        return

    errors = []
    failed_paths = set()
    total_duration = 0.0
    durations_by_file = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_audio = {executor.submit(read_audio, path[0]): path[0] for path in audio_paths}
        
        for idx, future in enumerate(as_completed(future_to_audio), start=1):
            audio_path = future_to_audio[future]
            try:
                error, duration = future.result()
                if error:
                    errors.append(error)
                    path, err_msg = error.split('\t', 1)
                    failed_paths.add(path)
                    print(f"[{idx}/{total_files}] Error reading: {path} | {err_msg}")
                else:
                    if args.calculate_duration:
                        total_duration += duration
                        durations_by_file.append((audio_path, duration))
                        print(f"[{idx}/{total_files}] Successfully read: {audio_path} (Duration: {format_duration(duration)})")
                    else:
                        print(f"[{idx}/{total_files}] Successfully read: {audio_path}")
            except Exception as exc:
                error_message = f"{audio_path}\tUnhandled exception: {str(exc)}"
                errors.append(error_message)
                failed_paths.add(audio_path)
                print(f"[{idx}/{total_files}] Exception reading {audio_path}: {exc}")

    if errors:
        try:
            with open(args.output_path, 'w') as error_file:
                for error in errors:
                    error_file.write(error + '\n')
            print(f"\nFinished processing. {len(errors)} errors logged to {args.output_path}.")
            
            # Update the TSV file to remove failed paths
            update_tsv_file(args.input_path, failed_paths, prefix)
        except Exception as e:
            print(f"Failed to write to the error log file: {str(e)}")
    else:
        print("\nFinished processing. No errors encountered.")

    if args.calculate_duration:
        print(f"\nTotal audio duration: {format_duration(total_duration)} (HH:MM:SS)")
        print(f"Total duration in seconds: {total_duration:.2f}")
        
        if args.duration_output:
            try:
                write_duration_report(args.duration_output, total_duration, durations_by_file)
                print(f"\nDuration information written to: {args.duration_output}")
            except Exception as e:
                print(f"Failed to write duration report: {str(e)}")

    if errors:
        error_types = [error.split('\t')[1] for error in errors]
        error_counter = Counter(error_types)
        print("\nError Summary:")
        for error_msg, count in error_counter.items():
            print(f"{error_msg}: {count} occurrence(s)")

if __name__ == "__main__":
    main()