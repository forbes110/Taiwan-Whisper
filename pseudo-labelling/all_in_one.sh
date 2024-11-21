#!/bin/bash
# TODO: note that this is different from tw_seperated
# init_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sample_1"   
init_dir="/mnt/home/ntuspeechlabtaipei1/soundon_separated"   
# init_dir="/mnt/home/ntuspeechlabtaipei1/tw_seperated/tw_seperated"  

# meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sample_2"   
meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sr_16k"  
pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label"
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"
minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection"
model_card=large-v2

# Format: YYYY-MM-DD HH:MM:SS
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# init_dir="/home/guest/b09705011/mnt/_dataset_meta"  
# meta_dir="/home/guest/b09705011/mnt/dataset_sr_16k"  
# pseudo_label_dir="/home/guest/b09705011/mnt/pseudo_label"
# data_pair_dir="/home/guest/b09705011/mnt/data_pair"
# metadata_dir="/home/guest/b09705011/mnt/metadata"
# minnan_dir="/home/guest/b09705011/mnt/minnan_detection"
# num_workers=2
# model_card=tiny

# # 1. bash resample.sh
# echo "Step 1 - Resample start: $(timestamp)"
# python3 resample.py \
#     --input "$init_dir" \
#     --max_workers 150 \
#     --invalid_channels ./invalid_channel.tsv \
#     --output_dir "$meta_dir" | tee resample.log
# echo "Step 1 - Resample end: $(timestamp)"
# echo "--------------------------------------------------------------------------------------------------------------------------------------------------------" | tee -a resample.log


# # 2. bash make_paths.sh
# python3 make_paths.py \
#     --root_dir "$meta_dir"

# 3. bash initial_inference.sh
echo "Step 3 - Initial inference start: $(timestamp)" | tee -a initial_inference.log
for audio_dir in "$meta_dir"/*/; do
    if [ -d "$audio_dir" ]; then
        echo "Start processing $(basename "$audio_dir") at $(timestamp)" | tee -a initial_inference.log
        python3 initial_inference.py \
            --dataset_path "$meta_dir/$(basename "$audio_dir").tsv" \
            --output_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
            --language zh \
            --log_progress True \
            --model_card $model_card \
            --compute_type default \
            --chunk_length 5 \
            --num_workers 8 \
            --repetition_penalty 10 \
            --word_timestamps True | tee -a initial_inference.log 
        echo "Complete processing $(basename "$audio_dir") at $(timestamp)" | tee -a initial_inference.log
    fi
done
echo "Step 3 - Initial inference end: $(timestamp)" | tee -a initial_inference.log
echo "----------------------------------------------------------------------------" | tee -a initial_inference.log

# 4. bash post_processing.sh
python3 simp2trad.py \
    --path "$pseudo_label_dir" \
    --output overwrite | tee post_processing.log

# 5. bash prepare_dataset.sh
for audio_dir in "$meta_dir"/*/; do
    echo "Start segmenting $audio_dir at $(timestamp)"
    python3 prepare_dataset.py \
        --audio_dir "$audio_dir" \
        --trans_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
        --segment_output_dir "$data_pair_dir/$(basename "$audio_dir")" \
        --nprocs 150 | tee -a prepare_dataset.log
done

# 6. bash minnan_detection.sh
for audio_dir in "$meta_dir"/*/; do
    echo "Start scanning $data_pair_dir/$(basename "$audio_dir") at $(timestamp)"
    python3 minnan_detection.py \
        --directory "$data_pair_dir/$(basename "$audio_dir")" \
        --csv_output_dir "$minnan_dir/$(basename "$audio_dir")" \
        --num_workers 8 \
        --to_remove | tee -a minnan_detection.log
done

# 7. bash gen_metadata.sh
for audio_dir in "$meta_dir"/*/; do
    python3 gen_metadata.py "$data_pair_dir/$(basename "$audio_dir")" \
        --valid-percent 0 \
        --dest "$metadata_dir" \
        --output_fname "$(basename "$audio_dir")" | tee -a gen_metadata.log
done


