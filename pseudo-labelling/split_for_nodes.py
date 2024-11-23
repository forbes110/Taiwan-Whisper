import pandas as pd
import math
import sys
import os

def split_csv(input_csv, num_splits, done_csv=None, output_prefix='node', output_suffix='_channels.csv'):
    """
    Splits the input CSV into multiple smaller CSV files after excluding channels in done_csv.

    Args:
        input_csv (str): Path to the input CSV file.
        num_splits (int): Number of splits to create.
        done_csv (str, optional): Path to the CSV file containing channels to exclude.
        output_prefix (str): Prefix for the output CSV files.
        output_suffix (str): Suffix for the output CSV files.
    """
    # Check if the input CSV exists
    if not os.path.isfile(input_csv):
        print(f"Error: The file '{input_csv}' does not exist.")
        sys.exit(1)
    
    # Read the input CSV file
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading '{input_csv}': {e}")
        sys.exit(1)
    
    # Ensure 'channels' column exists in input CSV
    if 'channel_name' not in df.columns:
        print(f"Error: The input CSV '{input_csv}' does not contain a 'channel_name' column.")
        sys.exit(1)
    
    # If done_csv is provided, read it and exclude those channels
    if done_csv:
        if not os.path.isfile(done_csv):
            print(f"Error: The done CSV file '{done_csv}' does not exist.")
            sys.exit(1)
        
        try:
            done_df = pd.read_csv(done_csv)
        except Exception as e:
            print(f"Error reading done CSV '{done_csv}': {e}")
            sys.exit(1)
        
        # Ensure 'channels' column exists in done_csv
        if 'channel_name' not in done_df.columns:
            print(f"Error: The done CSV '{done_csv}' does not contain a 'channel_name' column.")
            sys.exit(1)
        
        # Get the list of channels to exclude
        done_channels = set(done_df['channel_name'].dropna().astype(str))
        
        # Exclude done channels from the main dataframe
        initial_count = len(df)
        df = df[~df['channel_name'].astype(str).isin(done_channels)]
        excluded_count = initial_count - len(df)
        print(f"Excluded {excluded_count} channels from '{input_csv}' based on '{done_csv}'.")
    else:
        print("No '--done_csv' provided. Proceeding with all channels in the input CSV.")

    # Get the total number of channels after exclusion
    total_channels = len(df)
    print(f"Total channels to be split: {total_channels}")

    if total_channels == 0:
        print("No channels left to split after exclusion. Exiting.")
        sys.exit(0)
    
    # Calculate the number of rows per split
    rows_per_split = math.ceil(total_channels / num_splits)
    
    for i in range(num_splits):
        start_row = i * rows_per_split
        end_row = start_row + rows_per_split
        split_df = df.iloc[start_row:end_row]
        
        # Define output file name
        output_file = f"{output_prefix}_{i+1}{output_suffix}"
        
        # Write the split to a new CSV file
        try:
            split_df.to_csv(output_file, index=False)
            print(f"Created '{output_file}' with {len(split_df)} channels.")
        except Exception as e:
            print(f"Error writing '{output_file}': {e}")
    
    print("\nCSV splitting completed successfully.")

if __name__ == "__main__":
    import argparse

    # Default parameters
    DEFAULT_NUM_SPLITS = 4
    DEFAULT_INPUT_CSV = 'channels.csv'
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple smaller CSV files after excluding done channels.")
    parser.add_argument('input_csv', nargs='?', default=DEFAULT_INPUT_CSV,
                        help=f"Path to the input CSV file (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument('-n', '--num_splits', type=int, default=DEFAULT_NUM_SPLITS,
                        help=f"Number of splits to create (default: {DEFAULT_NUM_SPLITS})")
    parser.add_argument('--done_csv', type=str, default=None,
                        help="Path to the CSV file containing channels to exclude (optional)")
    parser.add_argument('-p', '--prefix', type=str, default='node',
                        help="Prefix for the output CSV files (default: 'node')")
    parser.add_argument('-s', '--suffix', type=str, default='_channels.csv',
                        help="Suffix for the output CSV files (default: '_channels.csv')")

    args = parser.parse_args()

    split_csv(args.input_csv, args.num_splits, args.done_csv, args.prefix, args.suffix)



# python3 split_for_nodes.py /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/all_channels.csv --done_csv /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/done_channel_names.csv















