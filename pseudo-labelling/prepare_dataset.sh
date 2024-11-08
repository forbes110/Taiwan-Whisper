#!/bin/bash

# python3 prepare_dataset.py \
#     --audio_dir /mnt/dataset_1T/tmp_dir/Gooaye_tmp \
#     --trans_dir /mnt/pseudo_label \
#     --segment_output_dir /mnt/data_pair \
#     --nprocs 8
    
python3 prepare_dataset.py \
    --audio_dir /mnt/dataset_1T/FTV_selected \
    --trans_dir /mnt/pseudo_label/FTV_selected \
    --segment_output_dir /mnt/data_pair/FTV_selected \
    --nprocs 8



# case for for each dir in meta_dir
# meta_dir="/path/to/your/meta_dir"  # Replace with your actual path

# for audio_dir in $meta_dir/*; do
#     python3 prepare_dataset.py \
#         --audio_dir "$audio_dir" \
#         --trans_dir /mnt/pseudo_label \
#         --segment_output_dir /mnt/data_pair/$(basename "$audio_dir") \
#         --nprocs 8
# done



