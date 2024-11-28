from datasets import load_from_disk
SAMPLE_RATE = 16000

# Load the saved dataset
# dataset = load_from_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ML2021_ASR_ST')
dataset = load_from_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ASCEND')

# To see what columns/features are available:
print(dataset.features)

# {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None), 'transcription': Value(dtype='string', id=None), 'translation': Value(dtype='string', id=None), 'file': Value(dtype='string', id=None)}

for instance in dataset:
    print(instance["id"])

#     # Create FLAC filename using client_id
#     flac_path = os.path.join(output_dir, f"CV16_{instance['client_id']}.flac")
    
#     # Save as FLAC
#     sf.write(flac_path, audio_array, SAMPLE_RATE)  # Note: This will be 48kHz unless we resample