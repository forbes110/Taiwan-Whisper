import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Filter TSV file based on Minnan detected files')
    parser.add_argument('channel_name', help='Name of the channel to process')
    args = parser.parse_args()

    try:
        # Read the CSV file containing Minnan detected files
        csv_path = f"/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection/{args.channel_name}/minnan_detected.csv"
        minnan_df = pd.read_csv(csv_path)

        # Extract relative paths
        paths_to_remove = set(minnan_df['audio_path'].apply(lambda x: '/'.join(x.split('/')[-2:])))

        # Read and filter TSV file
        tsv_path = f"/mnt/home/ntuspeechlabtaipei1/forbes/metadata/{args.channel_name}.tsv"
        with open(tsv_path, 'r') as f:
            tsv_lines = f.readlines()

        filtered_lines = [line for line in tsv_lines 
                         if not any(path in line for path in paths_to_remove)]

        # Write filtered content
        with open(tsv_path, 'w') as f:
            f.writelines(filtered_lines)

        print(f"Original lines: {len(tsv_lines)}")
        print(f"Remaining lines: {len(filtered_lines)}")
        print(f"Removed {len(tsv_lines) - len(filtered_lines)} lines")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()