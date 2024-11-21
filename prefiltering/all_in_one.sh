#!/bin/bash
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"
common_hallucination_dir="/mnt/home/ntuspeechlabtaipei1/forbes/common_hallucination_caught"
validator_card="openai/whisper-medium"
validator_inference_dir="/mnt/home/ntuspeechlabtaipei1/forbes/validator_inference"
cleaned_dir="/mnt/home/ntuspeechlabtaipei1/forbes/cleaned"
# for CPU
num_workers=150
threshold=0.6

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# for loop for all source of channels

# 1. bash common_hallucination_removal.sh
echo "Step 1 - Common Hallucination Removal: $(timestamp)" | tee -a common_hallucination_removal.log
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        filename=$(basename "$original_tsv".tsv)
        echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log
        output_dir="$common_hallucination_dir/$filename"
        python3 common_hallucination_removal.py \
            --original_tsv "$original_tsv" \
            --output_dir "$output_dir" \
            --execute_removal
        echo "Complete [rocessing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log  
    fi
done

# 2. bash validator_inference.sh
echo "Step 2 - Validator Inference start: $(timestamp)" | tee -a validator_inference.log
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        output_dir="$validator_inference_dir/$filename"
        echo "Start inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log
        python3 validator_inference.py \
            --manifest "$original_tsv" \
            --output_dir "$output_dir" \
            --validator "$validator_card" \
            --batch_size 64
        echo "Complete inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log 
    fi
done

# 3. bash elim_hallucination.sh 
echo "Step 3 - Hallucination elimination start: $(timestamp)" | tee -a elim_hallucination.log
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        hyps_dir="$validator_inference_dir/$filename"
        echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a elim_hallucination.log
        python elim_hallucination.py \
            --original_tsv "$original_tsv" \
            --type whisper \
            --hyps_txt "$hyps_dir"/elim_hallucination.txt \
            --output_dir "$cleaned_dir/$filename" \
            --threshold $threshold \
            --num_workers $num_workers \
            --phonemize \
            --mix_detection
        echo "Complete processing $(basename "$filename") at $(timestamp)" | tee -a elim_hallucination.log 
    fi
done