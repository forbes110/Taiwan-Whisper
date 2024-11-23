#!/bin/bash
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair_sample"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata_sample"

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

for audio_dir in "$data_pair_dir"/*/; do         
    python3 gen_metadata.py "$data_pair_dir/$(basename "$audio_dir")" \
        --valid-percent 0 \
        --dest "$metadata_dir" \
        --output_fname "$(basename "$audio_dir")" | tee -a gen_metadata.log
done
# /mnt/home/ntuspeechlabtaipei1/forbes/data_pair_sample/0fcba015-80a6-4aa0-8a65-24d71dd798f1