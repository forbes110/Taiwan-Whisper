"""
Given a dir which have a lot of subdirs, all the subdir represent a channel, in the subdir, there would be
tsv files which the name takes the form "cleaned-threshold-{ratio}-phonemized-mix_detection.tsv"
takes the form:
/mnt/home/ntuspeechlabtaipei1/forbes/data_pair/049c00d3-d675-461e-9a11-522694633cdb
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_10952800-11432160.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_13308640-13786400.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_3820640-4300320.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_12828640-13308640.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_7151200-7629280.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_12360160-12828640.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_2386720-2866720.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_4300320-4769120.flac

Note that another tsv may have different meta path at first line

Now we need to trace each tsv in the subdirs to merge them into a dataset called train_{ratio}.tsv

the input(as args) would be dir path, ratio, output_dir, workers
    parser = argparse.ArgumentParser(description='Merge TSV datasets from multiple channels')
    parser.add_argument('--dir_path', help='Root directory containing the channel subdirectories')
    parser.add_argument('--ratio', help='Threshold ratio to process')
    parser.add_argument('--output_dir', help='Output directory for merged dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')

output(to output_dir) would be 
1. train_{ratio}.tsv and 
2. train_{ratio}_info.txt that record the channel numbers in this dataset, and the corresponding channel names

can use multi-thread to make multi-worker merge them

called by the form:

python merge_cleaned_channels.py \
    --dir_path /mnt/home/ntuspeechlabtaipei1/forbes/cleaned_sample \
    --ratio 0.6 \
    --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train \
    --workers 180

"""
import argparse
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TSVMerger:
    def __init__(self, dir_path: str, ratio: str, output_dir: str, workers: int):
        self.dir_path = dir_path
        self.ratio = ratio
        self.output_dir = output_dir
        self.workers = workers
        self.channel_info = []
        self.output_lock = threading.Lock()
        self.meta_path = "/mnt/home/ntuspeechlabtaipei1/forbes/data_pair/"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize output files
        self.output_tsv = os.path.join(output_dir, f"train_{ratio}.tsv")
        self.output_info = os.path.join(output_dir, f"train_{ratio}_info.txt")
        
        # Initialize output TSV with meta path
        with open(self.output_tsv, 'w') as f:
            f.write(f"{self.meta_path}\n")
            
        # Clear info file
        open(self.output_info, 'w').close()

    def find_tsv_files(self) -> List[Tuple[str, str]]:
        """Find all matching TSV files in subdirectories."""
        tsv_files = []
        pattern = f"cleaned-threshold-{self.ratio}-phonemized-mix_detection.tsv"
        
        for root, _, _ in os.walk(self.dir_path):
            tsv_path = os.path.join(root, pattern)
            matches = glob.glob(tsv_path)
            
            if matches:
                channel_name = os.path.basename(root)
                tsv_files.append((matches[0], channel_name))
        
        logging.info(f"Found {len(tsv_files)} TSV files to process")
        return tsv_files

    def process_tsv_file(self, file_info: Tuple[str, str]) -> None:
        """Process a single TSV file."""
        tsv_path, channel_name = file_info
        
        try:
            with open(tsv_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                logging.warning(f"Empty file: {tsv_path}")
                return
                
            # Write to output TSV with lock to prevent concurrent writing
            with self.output_lock:
                with open(self.output_tsv, 'a') as out_f:
                    # Skip the first line (old meta path) and write the rest
                    out_f.writelines(lines[1:])
                
                # Store channel info for later
                self.channel_info.append(channel_name)
                logging.info(f"Processed channel: {channel_name}")
                
        except Exception as e:
            logging.error(f"Error processing {tsv_path}: {str(e)}")

    def write_info_file(self) -> None:
        """Write the channel information file."""
        with open(self.output_info, 'w') as f:
            f.write(f"Total channels: {len(self.channel_info)}\n")
            f.write("\nChannel names:\n")
            for idx, channel in enumerate(sorted(self.channel_info), 1):
                f.write(f"{idx}. {channel}\n")

    def merge(self) -> None:
        """Main method to merge all TSV files."""
        tsv_files = self.find_tsv_files()
        
        if not tsv_files:
            logging.error(f"No TSV files found matching pattern in {self.dir_path}")
            return
            
        logging.info(f"Starting merge process with {self.workers} workers")
        
        # Process files using thread pool
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            executor.map(self.process_tsv_file, tsv_files)
            
        # Write channel information
        self.write_info_file()
        
        logging.info(f"Merge completed. Output files written to {self.output_dir}")
        logging.info(f"Total channels processed: {len(self.channel_info)}")

def main():
    parser = argparse.ArgumentParser(description='Merge TSV datasets from multiple channels')
    parser.add_argument('--dir_path', help='Root directory containing the channel subdirectories')
    parser.add_argument('--ratio', help='Threshold ratio to process')
    parser.add_argument('--output_dir', help='Output directory for merged dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    
    args = parser.parse_args()
    
    merger = TSVMerger(
        dir_path=args.dir_path,
        ratio=args.ratio,
        output_dir=args.output_dir,
        workers=args.workers
    )
    
    merger.merge()

if __name__ == "__main__":
    main()
    
    
    
# python merge_cleaned_channels.py \
#     --dir_path /mnt/home/ntuspeechlabtaipei1/forbes/cleaned \
#     --ratio 0.6 \
#     --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train \
#     --workers 180