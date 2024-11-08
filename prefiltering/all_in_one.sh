#!/bin/bash
metadata_dir="/mnt/metadata"


# for loop for all source of channels

# 1. bash common_hallucination_removal.sh

for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        filename=$(basename "$original_tsv" .tsv)
        output_dir="/mnt/common_hallucination_caught/$filename"

        python3 common_hallucination_removal.py \
            --original_tsv "$original_tsv" \
            --output_dir "$output_dir" \
            --execute_removal
    fi
done

# 2. bash validator_inference.sh
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        output_dir="/mnt/validator_inference/$filename"
        python3 validator_inference.py \
            --manifest "$original_tsv" \
            --output_dir "$output_dir" \
            --validator openai/whisper-tiny \
            --batch_size 64
    fi
done

# 3. bash elim_hallucination.sh 
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        hyps_dir="/mnt/validator_inference/$filename"
        python elim_hallucination.py \
            --original_tsv "$original_tsv" \
            --type whisper \
            --hyps_txt "$hyps_dir"/validator_inference.txt \
            --output_dir /mnt/cleaned/$filename \
            --threshold 0.4 \
            --num_workers 16 \
            --phonemize \
            --mix_detection
    fi
done