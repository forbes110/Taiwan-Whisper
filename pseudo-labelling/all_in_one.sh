#!/bin/bash
# init_dir="/mnt/home/ntuspeechlabtaipei1/tw_separated/tw_separated"  
# TODO: 測試 tw_seperated 一小部分的轉檔


init_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_meta"  
meta_dir="/mnt/home/ntuspeechlabtaipei1/forbes/dataset_sr_16k"  
pseudo_label_dir="/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label"
data_pair_dir="/mnt/home/ntuspeechlabtaipei1/forbes/data_pair"
metadata_dir="/mnt/home/ntuspeechlabtaipei1/forbes/metadata"
minnan_dir="/mnt/home/ntuspeechlabtaipei1/forbes/minnan_detection"
# num_workers=20
# model_card=large-v2
num_workers=20
model_card=large-v2

# init_dir="/home/guest/b09705011/mnt/_dataset_meta"  
# meta_dir="/home/guest/b09705011/mnt/dataset_sr_16k"  
# pseudo_label_dir="/home/guest/b09705011/mnt/pseudo_label"
# data_pair_dir="/home/guest/b09705011/mnt/data_pair"
# metadata_dir="/home/guest/b09705011/mnt/metadata"
# minnan_dir="/home/guest/b09705011/mnt/minnan_detection"
# num_workers=2
# model_card=tiny

# 1. bash resample.sh
python3 resample.py \
    --input "$init_dir" \
    --max_workers 4 \
    --invalid_channels ./invalid_channel.tsv \
    --output_dir "$meta_dir" | tee resample.log

# 2. bash make_paths.sh
python3 make_paths.py \
    --root_dir "$meta_dir"

# 3. bash initial_inference.sh
for audio_dir in "$meta_dir"/*/; do
    if [ -d "$audio_dir" ]; then
        echo "Start Processing $(basename "$audio_dir")"
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
        echo "Complete Processing $(basename "$audio_dir")"
    fi
done
# wait  # Wait for all background processes to finish

# 4. bash post_processing.sh
python3 simp2trad.py \
    --path "$pseudo_label_dir" \
    --output overwrite | tee post_processing.log

# 5. bash prepare_dataset.sh
for audio_dir in "$meta_dir"/*/; do
    python3 prepare_dataset.py \
        --audio_dir "$audio_dir" \
        --trans_dir "$pseudo_label_dir/$(basename "$audio_dir")" \
        --segment_output_dir "$data_pair_dir/$(basename "$audio_dir")" \
        --nprocs $num_workers | tee -a prepare_dataset.log
done

# 6. bash minnan_detection.sh
for audio_dir in "$data"/*/; do
    python3 minnan_detection.py \
        --directory "$data_pair_dir/$(basename "$audio_dir")" \
        --csv_output_dir "$minnan_dir/$(basename "$audio_dir")" \
        --num_workers $num_workers \
        --to_remove | tee -a minnan_detection.log
done

# 7. bash gen_metadata.sh
for audio_dir in "$meta_dir"/*/; do
    python3 gen_metadata.py "$data_pair_dir/$(basename "$audio_dir")" \
        --valid-percent 0 \
        --dest "$metadata_dir" \
        --output_fname "$(basename "$audio_dir")" | tee -a gen_metadata.log
done


