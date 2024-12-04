#!/bin/bash

# meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/tw_16k"  
meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/soundon_16k"  

# pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label_tw"
pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label_soundon"

data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

CSV_FILE="/mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/done_pairs_channel.csv"


check_channel_status() {
    local channel="$1"
    # 簡化的 awk 命令：只檢查第一欄（整行）是否匹配頻道名稱
    # 如果找到匹配，輸出任意非空字符串（這裡用 "found"）
    awk -F',' -v channel="$channel" '$1 == channel {print "found"}' "$CSV_FILE"
}

mark_as_processed() {
    local channel="$1"
    # 確保頻道名稱還不在 CSV 中才添加
    if [ -z "$(check_channel_status "$channel")" ]; then
        echo "$channel" >> "$CSV_FILE"
        echo "Marked $channel as processed in $CSV_FILE at $(timestamp)" | tee -a prepare_dataset.log
    fi
}

# Step 4: Simplified to Traditional Chinese conversion
# for audio_dir in "$meta_dir"/*/; do
#     base_name=$(basename "$audio_dir")
#     echo "Start processing(simp to trad) $audio_dir at $(timestamp)" | tee -a post_processing.log
#     python3 simp2trad.py \
#         --path "$pseudo_label_dir/$base_name" \
#         --workers 180 \
#         --output overwrite 2>&1 | tee -a post_processing.log
# done

# Step 5: Overlap handling
# for audio_dir in "$meta_dir"/*/; do
#     base_name=$(basename "$audio_dir")
#     echo "Start handling $audio_dir at $(timestamp)" | tee -a overlap_handling.log
#     python3 overlap_handling.py \
#         --input_dir "$pseudo_label_dir/$base_name" \
#         --output_dir "$pseudo_label_dir/$base_name" \
#         --num_workers 180 2>&1 | tee -a overlap_handling.log
# done

#--------------------------------------------------------------------------------

# # Step 6: Prepare dataset

# Step 6: Prepare dataset with skipping logic for processed channels
echo "Starting dataset preparation at $(timestamp)" 2>&1 | tee -a prepare_dataset.log

# Main processing loop for dataset preparation
for audio_dir in "$meta_dir"/*/; do
    base_name=$(basename "$audio_dir")
    echo "Checking channel $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
    
    # Check if this channel has already been processed
    if [ -n "$(check_channel_status "$base_name")" ]; then
        echo "Skipping $base_name - already processed" | tee -a prepare_dataset.log
        continue
    fi
        
    # Verify that required directories exist before processing
    if [[ ! -d "$audio_dir" ]]; then
        echo "Error: Audio directory not found for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
        continue
    fi
    
    if [[ ! -d "$pseudo_label_dir/$base_name" ]]; then
        echo "Error: Pseudo label directory not found for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
        continue
    fi
    
    echo "Starting dataset preparation for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
    
    # Create output directory if it doesn't exist
    mkdir -p "$data_pair_dir/$base_name"
    
    if python3 prepare_dataset.py \
        --audio_dir "$audio_dir" \
        --trans_dir "$pseudo_label_dir/$base_name" \
        --segment_output_dir "$data_pair_dir/$base_name" \
        --nprocs 180 2>&1 | tee -a prepare_dataset.log; then
        
        # 檢查輸出目錄中是否有文件
        if [ -n "$(ls -A "$data_pair_dir/$base_name")" ]; then
            echo "Successfully completed dataset preparation for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
            # 標記為已處理
            mark_as_processed "$base_name"
        else
            echo "Warning: Output directory is empty for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
        fi
    else
        echo "Error: Dataset preparation failed for $base_name at $(timestamp)" 2>&1 | tee -a prepare_dataset.log
    fi
done


# Step 7: Generate metadata
for audio_dir in "$meta_dir"/*/; do
    base_name=$(basename "$audio_dir")
    python3 gen_metadata.py "$data_pair_dir/$base_name" \
        --valid-percent 0 \
        --dest "$metadata_dir" \
        --output_fname "$base_name" 2>&1 | tee -a gen_metadata.log
done