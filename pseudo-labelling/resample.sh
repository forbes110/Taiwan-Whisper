#!/bin/bash

# meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sr_16k"  
init_dir="/mnt/home/ntuspeechlabtaipei1/tw_separated"  
meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/tw_16k"  

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# 1. bash resample.sh
echo "Step 1 - Resample start at $(timestamp)" | tee -a resample.log
python3 resample.py \
    --input "$init_dir" \
    --max_workers 150 \
    --invalid_channels ./invalid_channel.tsv \
    --output_dir "$meta_dir" 2>&1 | tee resample.log
echo "Step 1 - Resample end at $(timestamp)"
echo "--------------------------------------------------------------------------------------------------------------------------------------------------------" | tee -a resample.log