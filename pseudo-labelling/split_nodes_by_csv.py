import os
import pandas as pd
import shutil
from pathlib import Path

def copy_tsv_files(metadata_dir, channel_name_csv, output_dir):
    """
    Copy TSV files from metadata_dir to output_dir if their channel names
    are present in the provided CSV file.
    
    Parameters:
        metadata_dir (str): Directory containing TSV files
        channel_name_csv (str): Path to CSV file containing channel names
        output_dir (str): Directory where matching files will be copied
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read channel names from CSV file and store them in a set for efficient lookup
    try:
        df = pd.read_csv(channel_name_csv)
        channel_names = set(df['channel_name'].str.strip().tolist())
    except KeyError:
        raise ValueError("CSV file must contain a 'channel name' column")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    
    # Convert paths to Path objects for easier manipulation and cross-platform compatibility
    metadata_path = Path(metadata_dir)
    output_path = Path(output_dir)
    
    # Keep track of statistics to provide feedback to the user
    files_processed = 0
    files_copied = 0
    
    # Walk through the metadata directory and all its subdirectories
    for root, _, files in os.walk(metadata_path):
        for file in files:
            # Check for TSV files (both .tsv and .TSV extensions)
            if file.lower().endswith('.tsv'):
                files_processed += 1
                
                # Extract channel name from file path
                # Assuming the channel name is the parent directory name
                channel_name = file.removesuffix('.tsv')
                print("channel_name:", channel_name)
                
                if channel_name in channel_names:
                    # Create the same directory structure in output_dir
                    relative_path = Path(root).relative_to(metadata_path)
                    new_dir = output_path / relative_path
                    new_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file while preserving metadata
                    source_file = Path(root) / file
                    destination_file = new_dir / file
                    shutil.copy2(source_file, destination_file)
                    files_copied += 1
    
    if files_processed == 0:
        print("Warning: No TSV files found in the specified directory.")
    
    return files_processed, files_copied

def main():
    """
    Main function to run the script with command line arguments.
    Provides a user-friendly interface for running the file copying utility.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy TSV files based on channel names from CSV')
    parser.add_argument('--metadata_dir', help='Directory containing TSV files')
    parser.add_argument('--channel_name_csv', help='CSV file containing channel names')
    parser.add_argument('--output_dir', help='Directory where matching files will be copied')
    
    args = parser.parse_args()
    
    try:
        print(f"Starting to process files...")
        files_processed, files_copied = copy_tsv_files(
            args.metadata_dir,
            args.channel_name_csv,
            args.output_dir
        )
        print(f"Processing complete!")
        print(f"Total files processed: {files_processed}")
        print(f"Files copied: {files_copied}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
    
# python3 split_nodes_by_csv.py \
#     --metadata_dir /mnt/home/ntuspeechlabtaipei1/forbes/metadata \
#     --channel_name_csv /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/node_0.csv \
#     --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/tw_metadata_node_0
        