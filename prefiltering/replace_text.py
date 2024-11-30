import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
from functools import partial

def process_file(filepath):
    """Process a single file to replace characters"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace('喫', '吃')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}", file=sys.stderr)
        return False

def scan_video_directory(video_dir):
    """Scan a video directory for txt files"""
    txt_files = []
    try:
        for file in os.listdir(video_dir):
            if file.endswith('.txt'):
                txt_files.append(os.path.join(video_dir, file))
    except Exception as e:
        print(f"Error scanning video directory {video_dir}: {str(e)}", file=sys.stderr)
    return txt_files

def main():
    search_path = '/mnt/home/ntuspeechlabtaipei1/forbes/data_pair'
    num_workers = 160
    
    print("---------------------------------Start---------------------------------")
    
    # Get channel directories
    channel_dirs = []
    with tqdm(desc="Scanning channel directories") as pbar:
        for channel in os.listdir(search_path):
            channel_path = os.path.join(search_path, channel)
            if os.path.isdir(channel_path):
                channel_dirs.append(channel_path)
                pbar.update(1)
    
    print(f"Found {len(channel_dirs)} channels")
    
    # Get all video directories
    video_dirs = []
    with tqdm(total=len(channel_dirs), desc="Scanning video directories") as pbar:
        for channel_dir in channel_dirs:
            try:
                for video in os.listdir(channel_dir):
                    video_path = os.path.join(channel_dir, video)
                    if os.path.isdir(video_path):
                        video_dirs.append(video_path)
            except Exception as e:
                print(f"Error accessing channel directory {channel_dir}: {str(e)}", file=sys.stderr)
            pbar.update(1)
    
    print(f"Found {len(video_dirs)} video directories")
    
    # Scan video directories in parallel
    txt_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(video_dirs), desc="Scanning for txt files") as pbar:
            # Submit all video directories for scanning
            futures = []
            for video_dir in video_dirs:
                future = executor.submit(scan_video_directory, video_dir)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                txt_files.extend(result)
    
    if not txt_files:
        print("No .txt files found!")
        return
    
    print(f"\nFound {len(txt_files)} files to process")
    
    # Process files in parallel with larger chunks
    chunk_size = 1000  # Process files in chunks to reduce overhead
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(txt_files), desc="Processing files") as pbar:
            futures = []
            # Process files in chunks
            for i in range(0, len(txt_files), chunk_size):
                chunk = txt_files[i:i + chunk_size]
                for filepath in chunk:
                    future = executor.submit(process_file, filepath)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)
            
            successful = sum(1 for future in futures if future.result())
    
    print(f"Completed! Successfully processed {successful} out of {len(txt_files)} files.")

if __name__ == '__main__':
    main()