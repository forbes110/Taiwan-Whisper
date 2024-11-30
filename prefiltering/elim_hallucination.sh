#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <i> <prefix>"
    echo "  <i>: Node number (0,1,2,3)"
    echo "  <prefix>: Data prefix (e.g., 'soundon' or 'tw')"
    exit 1
fi

# Set the input arguments
i="$1"      # Node number (0,1,2,3)
prefix="$2"  # Data prefix

# Validate prefix
if [[ "$prefix" != "soundon" && "$prefix" != "tw" ]]; then
    echo "Error: prefix must be either 'soundon' or 'tw'"
    exit 1
fi

# Directory paths using the prefix
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata/${prefix}_metadata_node_${i}"
common_hallucination_dir="/mnt/home/ntuspeechlabtaipei1/forbes/common_hallucination_caught"
validator_card="openai/whisper-base"
validator_inference_dir="/mnt/home/ntuspeechlabtaipei1/forbes/validator_inference"
cleaned_dir="/mnt/home/ntuspeechlabtaipei1/forbes/cleaned"
batch_size=64
# For CPU
num_workers=180
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

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <i> <prefix>"
    echo "  <i>: Node number (0,1,2,3)"
    echo "  <prefix>: Data prefix (e.g., 'soundon' or 'tw')"
    exit 1
fi

# Set the input arguments
i="$1"      # Node number (0,1,2,3)
prefix="$2"  # Data prefix

# Validate prefix
if [[ "$prefix" != "soundon" && "$prefix" != "tw" ]]; then
    echo "Error: prefix must be either 'soundon' or 'tw'"
    exit 1
fi

progress_file="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/elim_common_${prefix}.csv"

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Ensure the specified metadata directory exists
if [ ! -d "$metadata_dir" ]; then
    echo "Error: Directory $metadata_dir does not exist."
    exit 1
fi

# Create progress CSV if it doesn't exist
if [ ! -f "$progress_file" ]; then
    echo "channel_name,status,timestamp" > "$progress_file"
    echo "Created new progress tracking file: $progress_file"
fi

# Function to check if a channel has been processed
check_channel_status() {
    local channel="$1"
    # Using awk to search for the channel name in the CSV
    # Returns "completed" if found, empty string if not found
    awk -F',' -v channel="$channel" '$1 == channel && $2 == "completed" {print "completed"}' "$progress_file"
}

# Function to mark a channel as completed
mark_channel_completed() {
    local channel="$1"
    local current_time=$(timestamp)
    echo "$channel,completed,$current_time" >> "$progress_file"
}

# # Process each TSV file in the metadata directory
# echo "Starting hallucination removal process: $(timestamp)" | tee -a common_hallucination_removal.log

# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         channel_name=$(basename "$original_tsv" .tsv)
        
#         # Check if channel has already been processed
#         if [ -n "$(check_channel_status "$channel_name")" ]; then
#             echo "Skipping $channel_name - already processed" | tee -a common_hallucination_removal.log
#             continue
#         fi

#         echo "Start processing $channel_name at $(timestamp)" | tee -a common_hallucination_removal.log
#         output_dir="$common_hallucination_dir/$channel_name"
        
#         # Process the channel
#         if python3 common_hallucination_removal.py \
#             --original_tsv "$original_tsv" \
#             --output_dir "$output_dir" \
#             --num_threads 150 \
#             --execute_removal 2>&1 | tee -a common_hallucination_removal.log; then
            
#             # Mark channel as completed only if processing was successful
#             mark_channel_completed "$channel_name"
#             echo "Successfully completed processing $channel_name at $(timestamp)" | tee -a common_hallucination_removal.log
#         else
#             echo "Error processing $channel_name at $(timestamp)" | tee -a common_hallucination_removal.log
#         fi
#     fi
# done

# echo "Finished hallucination removal process: $(timestamp)" | tee -a common_hallucination_removal.log



# echo "Step 2 - Validator Inference start: $(timestamp)" | tee -a validator_inference.log
# channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/validated_channel_names.csv"
# channel_names_column="channel_name"

# for original_tsv in "$metadata_dir"/*.tsv; do
#     if [[ -f "$original_tsv" ]]; then
#         channel_name=$(basename "$original_tsv" .tsv)
#         output_dir="$validator_inference_dir/$channel_name"

#         # Skip if validator_inference.txt exists in the output directory
#         if [[ -f "$output_dir/validator_inference.txt" ]]; then
#             echo "Skipping ${channel_name} - already processed" | tee -a validator_inference.log
#             continue
#         fi

#         # Create output directory if it doesn't exist
#         mkdir -p "$output_dir"
        
#         # Log the start of inference for the current file
#         echo "Start inferencing ${channel_name} at $(timestamp)" 2>&1 | tee -a validator_inference.log
        
#         # Run the inference command and log both stdout and stderr
#         accelerate launch validator_inference.py \
#             --manifest "$original_tsv" \
#             --output_dir "$output_dir" \
#             --validator "$validator_card" \
#             --batch_size $batch_size 2>&1 | tee -a validator_inference.log
        
#         # Check if the previous command was successful
#         if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
#             echo "Error during inferencing $(basename "$channel_name") at $(timestamp)" | tee -a validator_inference.log
#         else
#             # Log the completion of inference for the current file
#             echo "Complete inferencing $(basename "$channel_name") at $(timestamp)" | tee -a validator_inference.log
#             echo "$base_name" >> "$channel_names_csv"
#         fi
#     fi
# done

# 3. bash elim_hallucination.sh 
echo "Step 3 - Hallucination elimination start: $(timestamp)" | tee -a "elim_hallucination_${i}.log"
for original_tsv in "$metadata_dir"/*.tsv; do
    if [[ -f "$original_tsv" ]]; then
        channel_name=$(basename "$original_tsv" .tsv)
        hyps_dir="$validator_inference_dir/$channel_name"
        echo "Start processing $(basename "$channel_name") at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
        python elim_hallucination.py \
            --original_tsv "$original_tsv" \
            --type whisper \
            --hyps_txt "$hyps_dir"/validator_inference.txt \
            --output_dir "$cleaned_dir/$channel_name" \
            --threshold $threshold \
            --num_workers $num_workers \
            --mix_detection \
            --phonemize 2>&1 | tee -a "elim_hallucination_${i}.log"
        echo "Complete processing $(basename "$channel_name") at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
    fi
    echo "Everything is done at $(timestamp)" | tee -a "elim_hallucination_${i}.log"
done

# bash all_in_one.sh 0 tw
# bash all_in_one.sh 1 tw
# bash all_in_one.sh 2 tw