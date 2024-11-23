# python3 minnan_detection.py \
#     --directory /mnt/data_pair/FTV_selected \
#     --csv_output_dir /mnt/minnan_detection/FTV_selected \
#     --num_workers 8
    # --to_remove



# python3 minnan_detection.py \
#     --directory /home/guest/b09705011/mnt/data_pair/FTV_selected_seq \
#     --csv_output_dir /home/guest/b09705011/mnt/minnan_detection/FTV_selected_seq \
#     --num_workers 1 \
#     --to_remove

#!/bin/bash

# Directories Configuration
meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sr_16k"  
pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label"
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"
minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection"
model_card=large-v2

# CSV Files Configuration
channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/done_channel_names.csv"
channel_names_column="channel_name"

# Timestamp Function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Load channel names into an array from the done_csv file
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

# Path to the current node's channels CSV (to be provided as an argument)
# Usage: ./minnan_detection.sh path_to_node_channels.csv
if [ $# -ne 1 ]; then
    echo "Usage: $0 path_to_node_channels.csv"
    exit 1
fi

node_channels_csv="$1"

# Check if the node_channels_csv exists
if [ ! -f "$node_channels_csv" ]; then
    echo "Error: The file '$node_channels_csv' does not exist."
    exit 1
fi

# Read channels from node_channels_csv into an array
mapfile -t node_channels < <(awk -F, '
BEGIN {header=1}
{
    if (header) {
        header = 0
        next
    }
    print $1  # Assuming the first column contains the channel names
}' "$node_channels_csv")

# Print total channels to be processed
total_node_channels=${#node_channels[@]}
echo "Total channels to be processed by this node: $total_node_channels"

# Loop through each channel in node_channels.csv
for channel in "${node_channels[@]}"; do
    # Check if the channel is already in done_csv
    if [[ " ${channel_names[*]} " == *" $channel "* ]]; then
        echo "Skipping $channel (already processed)" | tee -a minnan_detection.log
        continue  # Skip processing this channel
    fi

    # Construct the full path to the channel's data_pair directory
    full_path="$data_pair_dir/$channel"

    # Check if the directory exists
    if [ ! -d "$full_path" ]; then
        echo "Directory '$full_path' does not exist. Skipping." | tee -a minnan_detection.log
        continue
    fi

    # Log the start of processing
    echo "Start scanning $full_path at $(timestamp)" | tee -a minnan_detection.log

    # Run minnan_detection.py
    python3 minnan_detection.py \
        --directory "$full_path" \
        --csv_output_dir "$minnan_dir/$channel" \
        --num_workers 8 \
        --to_remove | tee -a minnan_detection.log

    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Processing completed for $channel. Adding to the done CSV." | tee -a minnan_detection.log
        # Append the channel to the done_csv
        echo "$channel" >> "$channel_names_csv"
    else
        echo "Processing failed for $channel. Skipping addition to the done CSV." | tee -a minnan_detection.log
    fi

    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------" | tee -a minnan_detection.log
done

echo "Min Nan detection process completed on $(timestamp)." | tee -a minnan_detection.log
