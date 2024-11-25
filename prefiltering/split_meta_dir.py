import os
import argparse
import shutil
from math import ceil

def split_dir(input_dir, num_nodes):
    """
    Split a directory containing CSV files into multiple subdirectories based on the number of nodes.
    Does not modify the original directory; files are copied instead of moved.
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Get all CSV files from the input directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.tsv')]
    total_files = len(all_files)

    if total_files == 0:
        print("No TSV files found in the input directory.")
        return

    # Calculate the number of files per node
    files_per_node = ceil(total_files / num_nodes)

    # Create subdirectories and distribute files among them
    for i in range(num_nodes):
        node_dir = os.path.join(input_dir, f"metadata_node_{i}")  # Create a subdirectory name
        os.makedirs(node_dir, exist_ok=True)  # Ensure the directory is created
        start_idx = i * files_per_node  # Starting index for file allocation
        end_idx = min(start_idx + files_per_node, total_files)  # Ending index
        for file in all_files[start_idx:end_idx]:
            shutil.copy2(os.path.join(input_dir, file), os.path.join(node_dir, file))  # Copy file to subdirectory
        print(f"Created {node_dir} with {end_idx - start_idx} files.")  # Output status for the current subdirectory

    print("Splitting completed.")  # Notify that the process is complete

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split a directory into multiple subdirectories.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing TSV files.")  # Input directory
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of splits.")  # Number of subdirectories
    args = parser.parse_args()

    # Call the split function
    split_dir(args.input_dir, args.num_nodes)


# python3 split_meta_dir.py /mnt/home/ntuspeechlabtaipei1/forbes/metadata --num_nodes 4