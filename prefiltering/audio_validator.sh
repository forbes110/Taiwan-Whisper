#!/bin/bash

# Check if node number argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <node_number>"
    echo "Example: $0 0"
    exit 1
fi

node_number="$1"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata/tw_metadata_node_${node_number}"
progress_file="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/filtered_metadata.csv"
failed_audios_file="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/failed_audios.txt"

# Check if the metadata directory exists
if [ ! -d "$metadata_dir" ]; then
    echo "Error: Directory $metadata_dir does not exist"
    exit 1
fi

# Create progress file if it doesn't exist
if [ ! -f "$progress_file" ]; then
    echo "filename,completion_time,node_number,num_failed" > "$progress_file"
fi

# Read validated channel names from CSV (skip header) and trim whitespace
channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/validated_channel_names.csv"
mapfile -t channel_names < <(tail -n +2 "$channel_names_csv" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

# Debug: Show loaded channel names info
echo "First 5 channel names:"
for i in {0..4}; do
    echo "Channel $i: '${channel_names[$i]}'"
done
echo "Total channels loaded: ${#channel_names[@]}"

# Process each TSV file
for original_tsv in "$metadata_dir"/*.tsv; do
    filename=$(basename "$original_tsv" .tsv)
    
    # Check if file has already been processed
    if grep -q "${filename}.tsv" "$progress_file"; then
        echo "Skipping ${filename}.tsv - already processed"
        continue
    fi
    
    # Improved channel name checking with trimmed strings
    found=0
    for channel in "${channel_names[@]}"; do
        if [[ "${channel}" == "${filename}" ]]; then
            echo "Skipping ${filename}.tsv - channel already validated"
            found=1
            break
        fi
    done
    
    if [[ $found -eq 1 ]]; then
        continue
    fi
    
    echo "Processing ${filename}.tsv from node $node_number..."
    
    # Clear the failed_audios file before processing new TSV
    # > "$failed_audios_file"
    
    # Run the validation script
    python3 audio_validator.py \
        --input_path "${original_tsv}" \
        --num_workers 180 \
        --output_path "$failed_audios_file"
    
    # Count number of failed files
    num_failed=$(wc -l < "$failed_audios_file")
    
    # Record completion time, filename, node number, and number of failed files
    completion_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${filename}.tsv,$completion_time,$node_number,$num_failed" >> "$progress_file"
    
    echo "Completed processing ${filename}.tsv at $completion_time (Failed files: $num_failed)"
done


# python3 audio_validator.py \
#     --input_path "/mnt/home/ntuspeechlabtaipei1/forbes/metadata/tw_metadata_node_1/57ETFN.tsv" \
#     --num_workers 180 \
#     --output_path "/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/failed_audios_temp.txt"