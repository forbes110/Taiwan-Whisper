import os
import csv

TARGET_SIZE_GB = 400  # Target size in GB
TARGET_SIZE_BYTES = TARGET_SIZE_GB * (1024 ** 3)  # Convert to bytes
MAX_FILE_SIZE_BYTES = 3.99 * (1024 ** 3)  # Maximum file size limit in bytes

def get_all_flac_files(directory):
    """Retrieve all .flac files and their sizes (in bytes) from the given directory."""
    flac_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                flac_files.append((file_path, file_size))
    return flac_files

def select_files_for_target_size(files, target_size):
    """Select files until the total size reaches the target size."""
    # Sort files by size in descending order
    files = sorted(files, key=lambda x: x[1], reverse=True)

    selected_files = []
    total_size = 0

    for file, size in files:
        if size > MAX_FILE_SIZE_BYTES:
            print(f"Skipping {file} (size: {size} bytes) - exceeds maximum file size limit.")
            continue  # Skip files that exceed the size limit
        if total_size + size > target_size:
            break  # Stop if adding this file exceeds the target size
        selected_files.append(file)  # Store the file path
        total_size += size

    return selected_files, total_size

def save_file_paths_to_csv(selected_files):
    """Save the selected file paths to a CSV file."""
    csv_path = "selected_file_paths.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Path"])  # Header row
        for file in selected_files:
            writer.writerow([file])  # Write file path

    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    directory = "/mnt/dataset_1T/FTV_News_flac"  # Target directory

    # Get all .flac files and their sizes
    all_files = get_all_flac_files(directory)

    # Select files to reach approximately 400 GB
    selected_files, total_size = select_files_for_target_size(all_files, TARGET_SIZE_BYTES)

    # Save the selected file paths to a CSV
    save_file_paths_to_csv(selected_files)

    # Calculate and display the total size of the selected files
    total_size_gb = total_size / (1024 ** 3)  # Convert to GB
    print(f"Total size of selected files: {total_size} bytes ({total_size_gb:.2f} GB)")
