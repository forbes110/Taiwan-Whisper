# #!/bin/bash
# data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
# metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"

# # Format: YYYY-MM-DD HH:MM:SS
# timestamp() {
#     date "+%Y-%m-%d %H:%M:%S"
# }

# for audio_dir in "$data_pair_dir"/*/; do         
#     python3 gen_metadata.py "$data_pair_dir/$(basename "$audio_dir")" \
#         --valid-percent 0 \
#         --dest "$metadata_dir" \
#         --output_fname "$(basename "$audio_dir")" | tee -a gen_metadata.log
# done
# # /mnt/home/ntuspeechlabtaipei1/forbes/data_pair_sample/0fcba015-80a6-4aa0-8a65-24d71dd798f1




#!/bin/bash

# Usage: ./make_paths.sh <csv_file_path>
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <csv_file_path>"
    exit 1
fi

csv_file="$1"  # CSV file path passed as an argument
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Check if the provided CSV file exists
if [ ! -f "$csv_file" ]; then
    echo "Error: CSV file $csv_file does not exist."
    exit 1
fi

# Extract the "channel_name" column from the CSV, excluding the header
channel_names=$(awk -F',' 'NR > 1 {print $1}' "$csv_file")

# Process only directories that match the "channel_name" column
for channel_name in $channel_names; do
    audio_dir="$data_pair_dir/$channel_name"
    if [ -d "$audio_dir" ]; then  # Check if the directory exists
        python3 gen_metadata.py "$audio_dir" \
            --valid-percent 0 \
            --dest "$metadata_dir" \
            --output_fname "$channel_name" | tee -a gen_metadata.log
    else
        echo "$(timestamp) - Warning: Directory $audio_dir does not exist" | tee -a gen_metadata.log
    fi
done


# python3 gen_metadata.py 57ETFN \
#     --valid-percent 0 \
#     --dest /mnt/home/ntuspeechlabtaipei1/forbes/metadata \
#     --output_fname 57ETFN | tee -a gen_metadata.log

find /mnt/home/ntuspeechlabtaipei1/forbes/metadata -type f -name "*.tsv" -empty -print
