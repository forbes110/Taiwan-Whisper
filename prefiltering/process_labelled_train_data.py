"""
Process Training set:
DaAi Drama & CV16(train+others)
"""

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
    """
    # Load dataset and ensure output directory exists
    dataset = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counter for generating sequential IDs
    current_id = 1
    
    # Calculate padding width based on dataset size for consistent ID formatting
    # This ensures all IDs will have the same number of digits (e.g., 001, 002, ...)
    id_padding = len(str(len(dataset)))
    
    records = []
    for instance in tqdm(dataset, desc=f"Processing {prefix.strip('_')} files"):
        # Generate or extract the ID based on idx_name
        if idx_name is None:
            # Generate a zero-padded sequential ID with prefix
            clean_prefix = prefix.rstrip('_')  # Remove trailing underscore if present
            idx = f"{clean_prefix}_{str(current_id).zfill(id_padding)}"
            current_id += 1
        else:
            # For existing IDs, still add the prefix if it's not already there
            raw_idx = instance[idx_name]
            clean_prefix = prefix.rstrip('_')
            if not raw_idx.startswith(f"{clean_prefix}_"):
                idx = f"{clean_prefix}_{raw_idx}"
            else:
                idx = raw_idx
                
        audio_path = os.path.join(output_dir, f"{idx}.flac")
        sf.write(audio_path, audio_array, sample_rate)
        
        """
        Concept: 
        
        1. Saved to flac first.
        
        2. Process text by <|0.00|>transcription<|x.xx|><|endfortext|> and saved to the same dir as audio(maybe at final_dataset) add a new function to process things above.
        """
        
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
        

        audio_path = os.path.join(output_dir, f"{idx}.flac")
        sf.write(audio_path, audio_array, sample_rate)
        
        
        """
        TODO: make tsv like
        /mnt/home/ntuspeechlabtaipei1/forbes/data_pair/0a7f107e-8015-403d-bed3-a1f9d7511bbf
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_13384000-13862400.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_7645120-8121920.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_8121920-8601600.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_25810880-26286080.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_33938880-34417280.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_13862400-14342240.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_5263040-5741120.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_35374400-35853440.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_21506240-21983680.flac
        10d16d61-f63f-4cdf-ab3f-766116753cb9/10d16d61-f63f-4cdf-ab3f-766116753cb9_9079040-9556480.flac
        ...
        With first row common metadata paths
        if replace the "flac" with "text" will find the text
        Note that there are no "prev", only current text, check the format of data_pair
        """
        
        
        # Store metadata including the ID used
        # records.append({
        #     'idx': idx,
        #     'transcription': transcription,
        #     'audio_path': audio_path
        # })
    
    # TODO: saved here, remember to add metapath at first line
    
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
def save_cv16(dataset_path: str, output_dir: str) -> None:
    """
    Converts CV16 dataset to FLAC files with metadata TSV.
    
    Args:
        dataset_path (str): Path to the saved ASCEND dataset
        output_dir (str): Directory where files will be saved
    """
    save_dataset_to_flac(
        idx_name = "client_id", # id for ASCEND
        transcription_name = "sentence", # transcription for ASCEND, 
        audio_array_name = "audio", 
        dataset_path = dataset_path,
        output_dir = output_dir,
        prefix = "CV16_"
    )
    