#!/bin/bash

meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/tw_16k"  
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"
minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection"


# meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sample_1_1"  
# data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair_sample"
# metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata_sample"
# minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection_sample"

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}


# Function to check if channel exists in CSV
channel_exists_in_csv() {
    local channel_name="$1"
    local csv_file="$2"
    # Skip header and check if channel_name exists in the CSV
    tail -n +2 "$csv_file" | cut -d',' -f1 | grep -q "^${channel_name}$"
    return $?
}

# Validate input arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <node_index>"
    exit 1
fi

NODE_INDEX="$1"
CSV_FILE="node_${NODE_INDEX}.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file $CSV_FILE not found"
    exit 1
fi

channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/done_channel_names.csv"
# channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/done_channel_names_toy.csv"
channel_names_column="channel_name"

# Load channel names into an array from the CSV file
mapfile -t channel_names < <(awk -F, -v col="$channel_names_column" '
BEGIN {header=1}  # Start by processing the header
{
    if (header) {
        for (i=1; i<=NF; i++) {  # Find the column index for the specified column name
            if ($i == col) {col_idx = i}
        }
        header = 0
    } else {
        print $col_idx  # Print the value from the desired column
    }
}' "$channel_names_csv")


for audio_dir in "$meta_dir"/*/; do
    base_name=$(basename "$audio_dir")

    # Check if the base name is in the list of channel names
    if [[ " ${channel_names[*]} " == *" $base_name "* ]]; then
        echo "Skipping $audio_dir (found in channel_names)" | tee -a minnan_detection.log
        continue  # Skip processing this directory
    fi

    if channel_exists_in_csv "$base_name" "$CSV_FILE"; then
        echo "Start scanning $data_pair_dir/$base_name at $(timestamp)" | tee -a minnan_detection.log
        python3 minnan_detection.py \
            --directory "$data_pair_dir/$base_name" \
            --csv_output_dir "$minnan_dir/$base_name" \
            --num_workers 8 \
            --metadata_dir "$metadata_dir" 2>&1 | tee -a minnan_detection.log
        
        if [ $? -eq 0 ]; then
            echo "Processing completed for $base_name" | tee -a minnan_detection.log
            echo "$base_name" >> "$channel_names_csv"
        else
            echo "Processing failed for $base_name" | tee -a minnan_detection.log
        fi
    fi
done

echo "All processing completed at $(timestamp)" | tee -a processing.log