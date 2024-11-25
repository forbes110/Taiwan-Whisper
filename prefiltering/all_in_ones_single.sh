# #!/bin/bash
# metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata" 
# # metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata_sample" 
# common_hallucination_dir="/mnt/home/ntuspeechlabtaipei1/forbes/common_hallucination_caught"
# validator_card="openai/whisper-base"
# validator_inference_dir="/mnt/home/ntuspeechlabtaipei1/forbes/validator_inference"
# cleaned_dir="/mnt/home/ntuspeechlabtaipei1/forbes/cleaned"
# # for CPU
# num_workers=150
# threshold=0.6

# # Format: YYYY-MM-DD HH:MM:SS
# timestamp() {
#     date "+%Y-%m-%d %H:%M:%S"
# }

# # for loop for all source of channels

# # 1. bash common_hallucination_removal.sh
# echo "Step 1 - Common Hallucination Removal: $(timestamp)" | tee -a common_hallucination_removal.log
# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         filename=$(basename "$original_tsv" .tsv)
#         echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log
#         output_dir="$common_hallucination_dir/$filename"
#         python3 common_hallucination_removal.py \
#             --original_tsv "$original_tsv" \
#             --output_dir "$output_dir" \
#             --execute_removal
#         echo "Complete processing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log  
#     fi
# done

# # 2. bash validator_inference.sh
# echo "Step 2 - Validator Inference start: $(timestamp)" | tee -a validator_inference.log
# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         filename=$(basename "$original_tsv" .tsv)
#         output_dir="$validator_inference_dir/$filename"
#         echo "Start inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log
#         accelerate launch --multi_gpu validator_inference.py \
#             --manifest "$original_tsv" \
#             --output_dir "$output_dir" \
#             --validator "$validator_card" \
#             --batch_size 64
#         echo "Complete inferencing $(basename "$filename" .tsv) at $(timestamp)" | tee -a validator_inference.log 
#     fi
# done

# # 3. bash elim_hallucination.sh 
# echo "Step 3 - Hallucination elimination start: $(timestamp)" | tee -a elim_hallucination.log
# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         filename=$(basename "$original_tsv" .tsv)
#         hyps_dir="$validator_inference_dir/$filename"
#         echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a elim_hallucination.log
#         python elim_hallucination.py \
#             --original_tsv "$original_tsv" \
#             --type whisper \
#             --hyps_txt "$hyps_dir"/validator_inference.txt \
#             --output_dir "$cleaned_dir/$filename" \
#             --threshold $threshold \
#             --num_workers $num_workers \
#             --phonemize \
#             --mix_detection
#         echo "Complete processing $(basename "$filename") at $(timestamp)" | tee -a elim_hallucination.log 
#     fi
# done

#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <i>"
    exit 1
fi

# Set the input argument as i
i="$1"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata/metadata_node_${i}"
common_hallucination_dir="/mnt/home/ntuspeechlabtaipei1/forbes/common_hallucination_caught_sample"
validator_card="openai/whisper-base"
validator_inference_dir="/mnt/home/ntuspeechlabtaipei1/forbes/validator_inference_sample"
cleaned_dir="/mnt/home/ntuspeechlabtaipei1/forbes/cleaned_sample"
batch_size=256
# For CPU
num_workers=150
threshold=0.6

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Ensure the specified metadata directory exists
if [ ! -d "$metadata_dir" ]; then
    echo "Error: Directory $metadata_dir does not exist."
    exit 1
fi

# # 1. bash common_hallucination_removal.sh
# echo "Step 1 - Common Hallucination Removal: $(timestamp)" | tee -a common_hallucination_removal.log
# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         filename=$(basename "$original_tsv" .tsv)
#         echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log
#         output_dir="$common_hallucination_dir/$filename"
#         python3 common_hallucination_removal.py \
#             --original_tsv "$original_tsv" \
#             --output_dir "$output_dir" \
#             --execute_removal
#         echo "Complete processing $(basename "$filename") at $(timestamp)" | tee -a common_hallucination_removal.log  
#     fi
# done

# # 2. bash validator_inference.sh
# # Start logging
# echo "Step 2 - Validator Inference start: $(timestamp)" | tee -a validator_inference.log

# # Loop through each TSV file in the metadata directory
# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         filename=$(basename "$original_tsv" .tsv)
#         output_dir="$validator_inference_dir/$filename"
        
#         # Create output directory if it doesn't exist
#         mkdir -p "$output_dir"
        
#         # Log the start of inference for the current file
#         echo "Start inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log
        
#         # Run the inference command and log both stdout and stderr
#         # WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 python3 validator_inference.py \
#         accelerate launch validator_inference.py \
#             --manifest "$original_tsv" \
#             --output_dir "$output_dir" \
#             --validator "$validator_card" \
#             --batch_size $batch_size 2>&1 | tee -a validator_inference.log
        
#         # Check if the previous command was successful
#         if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
#             echo "Error during inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log
#         else
#             # Log the completion of inference for the current file
#             echo "Complete inferencing $(basename "$filename") at $(timestamp)" | tee -a validator_inference.log
#         fi
#     fi
# done

# 3. bash elim_hallucination.sh 
echo "Step 3 - Hallucination elimination start: $(timestamp)" | tee -a "elim_hallucination_${i}.log"
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        filename=$(basename "$original_tsv" .tsv)
        hyps_dir="$validator_inference_dir/$filename"
        echo "Start processing $(basename "$filename") at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
        python elim_hallucination.py \
            --original_tsv "$original_tsv" \
            --type whisper \
            --hyps_txt "$hyps_dir"/validator_inference.txt \
            --output_dir "$cleaned_dir/$filename" \
            --threshold $threshold \
            --num_workers $num_workers #\
            # --mix_detection
            # --phonemize 
        echo "Complete processing $(basename "$filename") at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
    fi
    echo "Everything is done at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
done

# bash all_in_one.sh 0