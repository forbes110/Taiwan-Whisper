#!/bin/bash

meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sr_16k"  
pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label"
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair_sample"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata_sample"
minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection_sample"
model_card=large-v2

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}


# TODO: copy the transcription(tw_seperated & soundon) to pseudo_label_dir but del invalid_channel dir
# /mnt/home/ntuspeechlabtaipei1/soundon_transcription/tw_transcription/word
# del invalid_channel dir

# 4. bash post_processing.sh
python3 simp2trad.py \
    --path "$pseudo_label_dir" \
    --workers 150 \
    --output overwrite 2>&1 | tee -a post_processing.log

# 5. overlap handling
for audio_dir in "$meta_dir"/*/; do
    echo "Start handling $audio_dir at $(timestamp)" | tee -a overlap_handling.log
    python3 overlap_handling.py \
        --input_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
        --output_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
        --num_workers 2>&1 | tee -a overlap_handling.log
done


# 6. bash prepare_dataset.sh
for audio_dir in "$meta_dir"/*/; do
    echo "Start segmenting $audio_dir at $(timestamp)" | tee -a prepare_dataset.log
    python3 prepare_dataset.py \
        --audio_dir "$audio_dir" \
        --trans_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
        --segment_output_dir "$data_pair_dir/$(basename "$audio_dir")" \
        --nprocs 150 2>&1 | tee -a prepare_dataset.log
done

# 7. bash gen_metadata.sh
for audio_dir in "$meta_dir"/*/; do
    python3 gen_metadata.py "$data_pair_dir/$(basename "$audio_dir")" \
        --valid-percent 0 \
        --dest "$metadata_dir" \
        --output_fname "$(basename "$audio_dir")" 2>&1 | tee -a gen_metadata.log
done


# 8. bash minnan_detection.sh
# TODO: delete the corresponding path in metadata, with minnan_detection_temp
channel_names_csv="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/pseudo-labelling/done_channel_names.csv"
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

# Loop through directories in $meta_dir
for audio_dir in "$meta_dir"/*/; do
    # Get the base name of the directory
    base_name=$(basename "$audio_dir")
    
    # Check if the base name is in the list of channel names
    if [[ " ${channel_names[*]} " == *" $base_name "* ]]; then
        echo "Skipping $audio_dir (found in channel_names)" | tee -a minnan_detection.log
        continue  # Skip processing this directory
    fi

    # Process the directory if not skipped
    echo "Start scanning $data_pair_dir/$base_name at $(date +"%Y-%m-%d %H:%M:%S")" | tee -a minnan_detection.log
    python3 minnan_detection_temp.py \
        --directory "$data_pair_dir/$base_name" \
        --csv_output_dir "$minnan_dir/$base_name" \
        --num_workers 8 \
        --metadata_dir "$metadata_dir" 2>&1 | tee -a minnan_detection_temp.log
    
    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Processing completed for $audio_dir. Adding to the CSV."
        # Append the base name to the CSV
        echo "$base_name" >> "$channel_names_csv"
    else
        echo "Processing failed for $audio_dir. Skipping addition to the CSV."
    fi
done



