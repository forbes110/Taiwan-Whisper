#!/bin/bash
meta_dir="/mnt/dataset_1T"  # Replace with your actual path

# 1. bash resample.sh
python3 resample.py \
    --input "$meta_dir" \
    --max_workers 8

# 2. bash make_paths.sh
python3 make_paths.py \
    --root_dir /mnt/dataset_1T

# 3. bash initial_inference.sh
for audio_dir in "$meta_dir"/*; do
    python3 initial_inference.py \
        --dataset_path /mnt/dataset_1T/BabyBusTC/raw_data.tsv \
        --output_dir /mnt/pseudo_label/BabyBusTC_seq \
        --language zh \
        --log_progress True \
        --model_size tiny \
        --compute_type default \
        --chunk_length 5 \
        --num_workers 8 \
        --batch_size 16
done

# 4. bash post_processing.sh
python3 simp2trad.py \
    --path /mnt/pseudo_label/BabyBusTC\
    --output overwrite


# 5. bash prepare_dataset.sh
for audio_dir in "$meta_dir"/*; do
    python3 prepare_dataset.py \
        --audio_dir "$audio_dir" \
        --trans_dir /mnt/pseudo_label \
        --segment_output_dir /mnt/data_pair/$(basename "$audio_dir") \
        --nprocs 8
done

# 6. bash gen_metadata.sh
for audio_dir in "$meta_dir"/*; do
    python3 gen_metadata.py /mnt/data_pair/$(basename "$audio_dir") \
        --valid-percent 0 \
        --dest /mnt/metadata \
        --output_fname "$(basename "$audio_dir")"
done
