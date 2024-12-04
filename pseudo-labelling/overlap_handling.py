import argparse
import pandas as pd
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
from functools import partial

def setup_logger():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def process_time_segments(df):
    """
    Process time segments by sorting by start time and removing overlapping segments.
    
    Args:
        df: DataFrame with columns ['speaker', 'start', 'end', 'text']
            
    Returns:
        DataFrame with processed segments, sorted by start time with overlaps removed
    """
    # Convert time strings to float if they're strings with 's' suffix
    if isinstance(df['start'].iloc[0], str):
        df['start'] = df['start'].str.rstrip('s').astype(float)
        df['end'] = df['end'].str.rstrip('s').astype(float)
    
    # Sort by start time
    df = df.sort_values('start').reset_index(drop=True)
    
    # Remove overlapping segments
    segments_to_keep = []
    current_end = float('-inf')
    
    for idx, row in df.iterrows():
        if row['start'] >= current_end:
            segments_to_keep.append(idx)
            current_end = row['end']
    
    # Keep only non-overlapping segments
    df = df.loc[segments_to_keep].reset_index(drop=True)
    
    # Convert times back to string format with 's' suffix
    df['start'] = df['start'].astype(str) + 's'
    df['end'] = df['end'].astype(str) + 's'
    
    return df

def replace_chinese_characters(df):
    """
    Replace specific Chinese characters in the text column.
    
    Args:
        df: DataFrame with a 'text' column
            
    Returns:
        DataFrame with replaced characters
    """
    if 'text' in df.columns:
        df['text'] = df['text'].str.replace('喫', '吃')
    return df

def process_file(file_path, output_dir):
    """
    Process a single CSV file and save the results.
    
    Args:
        file_path: Path to input CSV file
        output_dir: Directory to save processed file
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Replace Chinese characters
        df = replace_chinese_characters(df)
        
        # Process the segments
        processed_df = process_time_segments(df)
        
        # Create output filename
        output_path = Path(output_dir) / f"{Path(file_path).name}"
        
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        
        return True, f"Successfully processed {file_path}"
        
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process time segments from CSV files in parallel')
    parser.add_argument('--input_dir', required=True, help='Directory containing input CSV files')
    parser.add_argument('--output_dir', required=True, help='Directory for output CSV files')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), 
                      help='Number of worker processes (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of CSV files
    input_dir = Path(args.input_dir)
    csv_files = list(input_dir.glob('*.csv'))
    
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    logger.info(f"Using {args.num_workers} workers")
    
    # Create partial function with fixed output_dir
    process_file_with_output = partial(process_file, output_dir=output_dir)
    
    # Process files in parallel with progress bar
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks and get futures
        future_to_file = {
            executor.submit(process_file_with_output, file_path): file_path
            for file_path in csv_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(csv_files), desc="Processing files") as pbar:
            for future in tqdm(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, message = future.result()
                    if success:
                        logger.info(message)
                    else:
                        logger.error(message)
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                finally:
                    pbar.update(1)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()