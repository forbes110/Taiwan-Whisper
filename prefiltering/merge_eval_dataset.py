from datasets import load_from_disk
import soundfile as sf
import os
import librosa
from tqdm import tqdm
import pandas as pd
from typing import List, Optional

from datasets import load_from_disk
import soundfile as sf
import os
import librosa
from tqdm import tqdm
import pandas as pd

def save_dataset_to_flac(
    dataset_path: str,
    output_dir: str,
    idx_name: str | None,
    transcription_name: str,
    audio_array_name: str = "audio",
    prefix: str = "",
    sample_rate: int = 16000
) -> None:
    """
    Converts audio data from a Hugging Face dataset to FLAC files and creates a metadata TSV.
    The function now handles cases where idx_name is None by generating sequential IDs with prefix.
    Also validates the uniqueness of idx_name column when provided.
    
    Args:
        dataset_path (str): Path to the saved Hugging Face dataset
        output_dir (str): Directory where FLAC files and metadata will be saved
        idx_name (str | None): Name of the field containing the unique identifier in the dataset.
                              If None, sequential IDs will be generated with prefix.
        transcription_name (str): Name of the field containing the text transcription
        audio_array_name (str): Name of the field containing the audio data (default: "audio")
        prefix (str): Prefix to add to output filenames (default: "")
        sample_rate (int): Target sampling rate in Hz (default: 16000)
    
    Returns:
        None: Files are saved to disk
    
    Raises:
        AssertionError: If idx_name is provided and contains duplicate values
    """
    # Load dataset and ensure output directory exists
    dataset = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check uniqueness of idx_name if provided
    if idx_name is not None:
        # Convert dataset column to list for duplicate checking
        idx_values = [instance[idx_name] for instance in dataset]
        # Find duplicate values using set comparison
        duplicates = [x for x in set(idx_values) if idx_values.count(x) > 1]
        
        if duplicates:
            error_msg = f"Column '{idx_name}' contains duplicate values: {duplicates}"
            print(f"錯誤：{error_msg}")
            assert len(duplicates) == 0, error_msg
        else:
            print(f"column '{idx_name}' is specified as idx column.")        
    
    # Initialize counter for generating sequential IDs
    current_id = 1
    
    # Calculate padding width based on dataset size for consistent ID formatting
    id_padding = len(str(len(dataset)))
    
    records = []
    for instance in tqdm(dataset, desc=f"Processing {prefix} files "):
        # Generate or extract the ID based on idx_name
        if idx_name is None:
            # Generate a zero-padded sequential ID with prefix
            idx = f"{prefix}_{str(current_id).zfill(id_padding)}"
            current_id += 1
        else:
            # For existing IDs, still add the prefix if it's not already there
            raw_idx = instance[idx_name]
            idx = f"{prefix}_{raw_idx}"
        
        # Extract audio and transcription data
        transcription = instance[transcription_name]
        audio_array = instance[audio_array_name]['array']
        original_sr = instance[audio_array_name]['sampling_rate']
        
        # Resample audio if needed
        if original_sr != sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=original_sr, 
                target_sr=sample_rate
            )
        
        # Create and save FLAC file with the ID
        audio_path = os.path.join(output_dir, f"{idx}.flac")
        sf.write(audio_path, audio_array, sample_rate)
        
        # Store metadata including the ID used
        records.append({
            'idx': idx,
            'transcription': transcription,
            'audio_path': audio_path
        })
    
    # Save metadata TSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'metadata.tsv'), 
              sep='\t', 
              index=False, 
              encoding='utf-8-sig')
    
    # Print summary of processing
    print(f"\nProcessing complete:")
    print(f"- Total files processed: {len(records)}")
    print(f"- Files saved to: {output_dir}")
    print(f"- Metadata saved to: {os.path.join(output_dir, 'metadata.tsv')}")
    
# Example usage for cv16 (you can create similar functions for other datasets):
def save_cv16(dataset_path: str, output_dir: str, prefix: str) -> None:
    """
    Converts CV16 dataset to FLAC files with metadata TSV.
    
    Args:
        dataset_path (str): Path to the saved ASCEND dataset
        output_dir (str): Directory where files will be saved
    """
    save_dataset_to_flac(
        idx_name = None, # id for ASCEND
        transcription_name = "sentence", # transcription for ASCEND, 
        audio_array_name = "audio", 
        dataset_path = dataset_path,
        output_dir = output_dir,
        prefix = prefix
    )
    
def save_minnan_sentence(dataset_path: str, output_dir: str, prefix: str) -> None:
    """
    Converts CV16 dataset to FLAC files with metadata TSV.
    
    Args:
        dataset_path (str): Path to the saved ASCEND dataset
        output_dir (str): Directory where files will be saved
    """
    save_dataset_to_flac(
        idx_name = None, # id for ASCEND
        transcription_name = "chinese", # transcription for ASCEND, 
        audio_array_name = "audio", 
        dataset_path = dataset_path,
        output_dir = output_dir,
        prefix = prefix
    )

def save_ML(dataset_path: str, output_dir: str, prefix: str) -> None:
    """
    Converts ML2021 dataset to FLAC files with metadata TSV. This function uses
    auto-generated sequential IDs since the dataset doesn't have unique identifiers.
    
    Args:
        dataset_path (str): Path to the saved ML2021 dataset
        output_dir (str): Directory where files will be saved
    """
    save_dataset_to_flac(
        dataset_path=dataset_path,
        output_dir=output_dir,
        idx_name=None,                  # This will trigger sequential ID generation
        transcription_name="transcription",
        audio_array_name="audio",
        prefix=prefix
    )
    
# Example of how you could create an ASCEND version:
def save_ascend(dataset_path: str, output_dir: str, prefix: str) -> None:
    """
    Converts ASCEND dataset to FLAC files with metadata TSV. This function is specifically
    configured for the ASCEND dataset structure.
    
    Args:
        dataset_path (str): Path to the saved ASCEND dataset
        output_dir (str): Directory where files will be saved
    """
    save_dataset_to_flac(
        dataset_path=dataset_path,
        output_dir=output_dir,
        idx_name="id",                  # Assuming ASCEND uses 'id' for unique identifiers
        transcription_name="transcription",      # Assuming ASCEND uses 'transcription' for transcriptions
        audio_array_name="audio",       # Standard audio field name
        prefix=prefix               # Adding ASCEND prefix to output files
    )
    

def merge_tsv_files(
    tsv_files: List[str],
    output_path: str,
    check_duplicates: bool = True,
    duplicate_cols: Optional[List[str]] = "idx"
) -> pd.DataFrame:
    """
    Merge multiple TSV files with identical column structure into a single TSV file.
    
    Args:
        tsv_files (List[str]): List of paths to TSV files to merge
        output_path (str): Path where the merged TSV will be saved
        check_duplicates (bool): Whether to check for and report duplicate entries
        duplicate_cols (List[str]): Columns to check for duplicates. If None, uses all columns
    
    Returns:
        pd.DataFrame: The merged dataframe
        
    Example:
        tsv_files = [
            "path/to/ML2021/metadata.tsv",
            "path/to/CV16/metadata.tsv",
            "path/to/ASCEND/metadata.tsv"
        ]
        merged_df = merge_tsv_files(tsv_files, "path/to/output/merged_metadata.tsv")
    """
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Read each TSV file
    for file_path in tsv_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue
            
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
        print(f"Reading {file_path}: {len(df)} rows")
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid TSV files were found to merge!")
    
    # Verify all dataframes have the same columns
    base_columns = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        if set(df.columns) != base_columns:
            raise ValueError(
                f"File {tsv_files[i]} has different columns than {tsv_files[0]}.\n"
                f"Expected: {base_columns}\n"
                f"Found: {set(df.columns)}"
            )
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Check for duplicates if requested
    if check_duplicates:
        cols_to_check = duplicate_cols if duplicate_cols else merged_df.columns
        duplicates = merged_df.duplicated(subset=cols_to_check, keep=False)
        if duplicates.any():
            print("\nWarning: Found duplicate entries:")
            print(merged_df[duplicates].sort_values(by=cols_to_check))
            print(f"\nTotal duplicates: {duplicates.sum()}")
    
    # Save merged dataframe
    merged_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8-sig')
    
    print(f"\nMerge complete:")
    print(f"- Total rows in merged file: {len(merged_df)}")
    print(f"- Merged file saved to: {output_path}")
    
    return merged_df
    
    
if __name__ == "__main__":
    
    # VALID & TEST set
    save_ML(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ML2021_ASR_ST',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ML2021_EVAL',
        prefix='ML2021_EVAL'
    )
    
    save_ML(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ML2021_ASR_ST',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ML2021_TEST',
        prefix='ML2021_TEST'
    )
    
    save_cv16(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/CV16',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/CV16_EVAL',
        prefix='CV16_EVAL'
    )
    
    save_cv16(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/CV16',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/CV16_TEST',
        prefix='CV16_TEST'
    )
    
    save_ascend(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ASCEND',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ASCEND_EVAL',
        prefix='ASCEND_EVAL'
    )
    save_ascend(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ASCEND',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ASCEND_TEST',
        prefix='ASCEND_TEST'
    )
    
    # TRAINING set
    save_cv16(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_train/CV17_train_minnan',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV17_TRAIN_MINNAN',
        prefix='CV17_TRAIN_MINNAN'
    )
    save_cv16(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_train/CV16_train',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV16_TRAIN',
        prefix='CV16_TRAIN'
    )
    save_cv16(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_train/CV16_other',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV16_OTHER',
        prefix='CV16_OTHER'
    )
    save_minnan_sentence(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_train/sentences_minnan',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/MINNAN',
        prefix='MINNAN'
    )
    
    # Merge
    # test_sets = [
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ASCEND_TEST/metadata.tsv",
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/CV16_TEST/metadata.tsv",
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ML2021_TEST/metadata.tsv"
    # ]
    # valid_sets = [
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ASCEND_EVAL/metadata.tsv",
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/CV16_EVAL/metadata.tsv",
    #     "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ML2021_EVAL/metadata.tsv"
    # ]
    
    # merge_tsv_files(tsv_files=test_sets, output_path="/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ACM_TEST.tsv")
    # merge_tsv_files(tsv_files=valid_sets, output_path="/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ACM_EVAL.tsv")