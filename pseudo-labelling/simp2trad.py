# TODO: make psuedo label from simplified chinese to traditional chinese

import os
import sys
import argparse
from opencc import OpenCC
import chardet

# Define the file extensions to process
TEXT_FILE_EXTENSIONS = {
    '.csv'
}

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        raw = f.read(100000)  # Read first 100KB
    result = chardet.detect(raw)
    return result['encoding']

def convert_file(file_path, converter, overwrite=True):
    """Convert Simplified Chinese to Traditional Chinese in a single file."""
    try:
        encoding = detect_encoding(file_path)
        if encoding is None:
            print(f"Skipping {file_path}: Unable to detect encoding.")
            return

        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        converted = converter.convert(content)

        if overwrite:
            with open(file_path, 'w', encoding=encoding, errors='ignore') as f:
                f.write(converted)
            print(f"Converted and overwritten: {file_path}")
        else:
            new_file = f"{file_path}.trad"
            with open(new_file, 'w', encoding=encoding, errors='ignore') as f:
                f.write(converted)
            print(f"Converted and saved as: {new_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, converter, overwrite=True):
    """Recursively process all text files in a directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in TEXT_FILE_EXTENSIONS:
                file_path = os.path.join(root, file)
                convert_file(file_path, converter, overwrite)

def main():
    parser = argparse.ArgumentParser(
        description="Convert Simplified Chinese text to Traditional Chinese in files or directories."
    )
    parser.add_argument(
        '--path',
        type=str,
        help="Path to the file or directory to convert."
    )
    parser.add_argument(
        '--output',
        choices=['overwrite', 'new'],
        default='overwrite',
        help="Choose to overwrite the original files or create new ones with a '.trad' extension. Default is 'overwrite'."
    )
    args = parser.parse_args()

    path = args.path
    overwrite = args.output == 'overwrite'

    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        sys.exit(1)

    # Initialize OpenCC converter for Simplified to Traditional
    converter = OpenCC('s2t')  # 's2t' stands for Simplified to Traditional

    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in TEXT_FILE_EXTENSIONS:
            convert_file(path, converter, overwrite)
        else:
            print(f"Skipping {path}: Unsupported file extension.")
    elif os.path.isdir(path):
        process_directory(path, converter, overwrite)
    else:
        print(f"Error: The path '{path}' is neither a file nor a directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
