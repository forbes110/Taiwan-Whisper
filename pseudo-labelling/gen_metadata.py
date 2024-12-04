import os
from datetime import datetime
import argparse
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_tsv_files(dir_path: str) -> List[str]:
    """Get all TSV files in the specified directory"""
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

def process_tsv_file(tsv_file: str) -> Tuple[str, List[str]]:
    """
    Process a single TSV file and return the processed file paths
    
    Returns:
        Tuple[str, List[str]]: (channel_name, list of processed paths)
    """
    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            logging.warning(f"Empty file: {tsv_file}")
            return os.path.splitext(os.path.basename(tsv_file))[0], []
            
        # Extract channel name from first line
        first_line = lines[0].strip()
        data_pair_index = first_line.find('data_pair/')
        if data_pair_index == -1:
            logging.error(f"Cannot find 'data_pair/' in file: {tsv_file}")
            return os.path.splitext(os.path.basename(tsv_file))[0], []
            
        channel = first_line[data_pair_index + len('data_pair/'):].strip()
        
        # Process remaining lines
        processed_paths = []
        for line in lines[1:]:
            paths = line.strip().split()
            processed_line = ' '.join(f"{channel}/{path}" for path in paths)
            if processed_line:
                processed_paths.append(processed_line)
                
        return os.path.splitext(os.path.basename(tsv_file))[0], processed_paths
    except Exception as e:
        logging.error(f"Error processing file {tsv_file}: {str(e)}")
        return os.path.splitext(os.path.basename(tsv_file))[0], []

def merge_channels(dir_path: str, output_dir: str, workers: int):
    """Merge multiple TSV files using multi-threading"""
    os.makedirs(output_dir, exist_ok=True)
    
    tsv_files = find_tsv_files(dir_path)
    logging.info(f"Found {len(tsv_files)} TSV files to process")
    
    # Collect processing results and statistics
    all_processed_lines = []
    channel_stats = {}
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(process_tsv_file, tsv_file): tsv_file 
                         for tsv_file in tsv_files}
        
        for future in future_to_file:
            try:
                channel_name, processed_lines = future.result()
                all_processed_lines.extend(processed_lines)
                channel_stats[channel_name] = len(processed_lines)
            except Exception as e:
                logging.error(f"Error executing task: {str(e)}")
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"train_{timestamp}.tsv")
    info_file = os.path.join(output_dir, f"train_{timestamp}_info.txt")
    
    # Write merged file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("/mnt/home/ntuspeechlabtaipei1/forbes/data_pair\n")
        for line in all_processed_lines:
            f.write(f"{line}\n")
    
    # Write info file
    with open(info_file, 'w', encoding='utf-8') as f:
        total_lines = sum(channel_stats.values())
        f.write(f"Merge Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Files: {len(channel_stats)}\n")
        f.write(f"Total Lines: {total_lines}\n\n")
        
        f.write("File Statistics:\n")
        f.write("-" * 50 + "\n")
        for channel, line_count in sorted(channel_stats.items()):
            f.write(f"File: {channel}.tsv\n")
            f.write(f"Lines: {line_count}\n")
            f.write("-" * 50 + "\n")
    
    logging.info(f"Merge completed, output file: {output_file}")
    logging.info(f"Statistics written to: {info_file}")
    logging.info(f"Total processed lines: {len(all_processed_lines)}")

def main():
    parser = argparse.ArgumentParser(description='Merge TSV datasets from multiple channels')
    parser.add_argument('--dir_path', required=True, help='Root directory containing channel subdirectories')
    parser.add_argument('--output_dir', required=True, help='Output directory for merged dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    
    args = parser.parse_args()
    merge_channels(args.dir_path, args.output_dir, args.workers)

if __name__ == '__main__':
    main()
