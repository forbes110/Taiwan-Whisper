"""
Given a dir which have a lot of subdirs, all the subdir represent a channel, in the subdir, there would be
tsv files which the name takes the form "cleaned-threshold-{ratio}-phonemized-mix_detection.tsv"
takes the form:

the first one
/mnt/home/ntuspeechlabtaipei1/forbes/data_pair/049c00d3-d675-461e-9a11-522694633cdb
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_10952800-11432160.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_13308640-13786400.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_3820640-4300320.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_12828640-13308640.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_7151200-7629280.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_12360160-12828640.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_2386720-2866720.flac
8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_4300320-4769120.flac

the second one
/mnt/home/ntuspeechlabtaipei1/forbes/data_pair/0bf5baa9-aeff-4d6d-8526-a25b8b597f48
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_40000320-40477600.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_24243840-24719360.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_40477600-40952800.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_56199680-56676000.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_46205600-46676640.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_30434880-30914240.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_35694240-36172640.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_55722240-56199680.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_53821120-54300160.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_44775360-45251680.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_28532800-29010720.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_22816320-23294240.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_45251680-45729440.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_53341440-53821120.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_48587200-49064960.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_33785280-34264640.flac
fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_34743360-35214720.flac


Note that another tsv may have different meta path at first line
so we use the common "/mnt/home/ntuspeechlabtaipei1/forbes/data_pair" as prefix and add the "049c00d3-d675-461e-9a11-522694633cdb" part of /mnt/home/ntuspeechlabtaipei1/forbes/data_pair/049c00d3-d675-461e-9a11-522694633cdb
back to the path 8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_10952800-11432160.flac
so if there are only 2 subdir 049c00d3-d675-461e-9a11-522694633cdb & 0bf5baa9-aeff-4d6d-8526-a25b8b597f48

there sould be:
/mnt/home/ntuspeechlabtaipei1/forbes/data_pair
049c00d3-d675-461e-9a11-522694633cdb/8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_10952800-11432160.flac
049c00d3-d675-461e-9a11-522694633cdb/8a87eef9-24b0-4293-bdce-3c0ef5453637/8a87eef9-24b0-4293-bdce-3c0ef5453637_13308640-13786400.flac
...
0bf5baa9-aeff-4d6d-8526-a25b8b597f48/fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_40000320-40477600.flac
0bf5baa9-aeff-4d6d-8526-a25b8b597f48/fc08880e-0e6f-4783-b4f5-1aac4236ce62/fc08880e-0e6f-4783-b4f5-1aac4236ce62_24243840-24719360.flac
...





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
    --dir_path /mnt/home/ntuspeechlabtaipei1/forbes/cleaned \
    --ratio 0.6 \
    --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train \
    --workers 180

"""

import os
from datetime import datetime
import argparse
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

def find_tsv_files(dir_path: str, ratio: str) -> List[str]:
    """Find all TSV files matching the pattern in the directory and its subdirectories."""
    pattern = f"cleaned-threshold-{ratio}-phonemized-mix_detection.tsv"
    tsv_files = []
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file == pattern:
                tsv_files.append(os.path.join(root, file))
    
    return tsv_files

def process_tsv_file(args: Tuple[str, str, str]) -> Tuple[str, List[str]]:
    """Process a single TSV file and return the channel name and processed lines."""
    tsv_file, data_pair_prefix, channel_dir = args
    processed_lines = []
    channel_name = os.path.basename(os.path.dirname(tsv_file))
    
    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Skip the first line (header/meta path)
        for line in lines[1:]:
            line = line.strip()
            if line:
                # Extract the relative path part
                path = line.split(data_pair_prefix)[-1].lstrip('/')
                # Prepend the channel directory name
                full_path = os.path.join(channel_dir, path)
                processed_lines.append(full_path)
                
        return channel_name, processed_lines
    except Exception as e:
        logging.error(f"Error processing file {tsv_file}: {str(e)}")
        return channel_name, []

def merge_channels(dir_path: str, ratio: str, output_dir: str, workers: int):
    """Merge TSV files from multiple channels using multi-threading."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TSV files
    tsv_files = find_tsv_files(dir_path, ratio)
    if not tsv_files:
        logging.error(f"No TSV files found matching ratio {ratio}")
        return
    
    logging.info(f"Found {len(tsv_files)} TSV files to process")
    
    # Common prefix for data pair paths
    data_pair_prefix = "/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
    
    # Process files using thread pool
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Create tasks for each TSV file
        future_to_file = {
            executor.submit(
                process_tsv_file, 
                (tsv_file, data_pair_prefix, os.path.basename(os.path.dirname(tsv_file)))
            ): tsv_file 
            for tsv_file in tsv_files
        }
        
        # Collect results as they complete
        for future in future_to_file:
            try:
                channel_name, processed_lines = future.result()
                results.append((channel_name, processed_lines))
            except Exception as e:
                logging.error(f"Error processing task: {str(e)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Write merged dataset
    output_file = os.path.join(output_dir, f"train_{ratio}_{timestamp}.tsv")
    info_file = os.path.join(output_dir, f"train_{ratio}_{timestamp}_info.txt")
    
    # Write the merged TSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the data pair prefix as the first line
        f.write(f"{data_pair_prefix}\n")
        # Write all processed lines
        for _, lines in results:
            for line in lines:
                f.write(f"{line}\n")
    
    # Write the info file
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Total channels: {len(results)}\n\n")
        f.write("Channel names:\n")
        for channel_name, _ in sorted(results, key=lambda x: x[0]):
            f.write(f"{channel_name}\n")
    
    logging.info(f"Merged dataset written to {output_file}")
    logging.info(f"Channel information written to {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge TSV datasets from multiple channels')
    parser.add_argument('--dir_path', help='Root directory containing the channel subdirectories')
    parser.add_argument('--ratio', help='Threshold ratio to process')
    parser.add_argument('--output_dir', help='Output directory for merged dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    
    args = parser.parse_args()
    
    merge_channels(args.dir_path, args.ratio, args.output_dir, args.workers)

if __name__ == '__main__':
    main()
    
    
    
# No GPU needed
# python merge_cleaned_channels.py \
#     --dir_path /mnt/home/ntuspeechlabtaipei1/forbes/cleaned \
#     --ratio 0.6 \
#     --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train \
#     --workers 180