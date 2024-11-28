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
    The function now handles cases where idx_name is None by generating sequential IDs.
    
    Args:
        dataset_path (str): Path to the saved Hugging Face dataset
        output_dir (str): Directory where FLAC files and metadata will be saved
        idx_name (str | None): Name of the field containing the unique identifier in the dataset.
                              If None, sequential IDs will be generated.
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
            # Generate a zero-padded sequential ID
            idx = str(current_id).zfill(id_padding)
            current_id += 1
        else:
            idx = instance[idx_name]
        
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
        
        # Create and save FLAC file with the ID (generated or extracted)
        audio_path = os.path.join(output_dir, f"{prefix}{idx}.flac")
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

def save_ML(dataset_path: str, output_dir: str) -> None:
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
        prefix="ML2021_"
    )
    
    
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
    
# Example of how you could create an ASCEND version:
def save_ascend(dataset_path: str, output_dir: str) -> None:
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
        prefix="ASCEND_"               # Adding ASCEND prefix to output files
    )
    
if __name__ == "__main__":
    # save_ML(
    #     dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ML2021_ASR_ST',
    #     output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ML2021'
    # )
    
    # save_ML(
    #     dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ML2021_ASR_ST',
    #     output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ML2021'
    # )
    
    # save_cv16(
    #     dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/CV16',
    #     output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/CV16'
    # )
    
    # save_cv16(
    #     dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/CV16',
    #     output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/CV16'
    # )
    
    save_ascend(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ASCEND',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/valid/ASCEND'
    )
    save_ascend(
        dataset_path='/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ASCEND',
        output_dir='/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/test/ASCEND'
    )
    
    