#!/bin/bash

# for audio_dir in "$meta_dir"/*; do
#     python3 gen_metadata.py "$(basename "$audio_dir")" \
#         --valid-percent 0 \
#         --dest /mnt \
#         --output_fname "$meta_dir/$(basename "$audio_dir")"
# done
meta_dir="/mnt/data_pair"

python3 gen_metadata.py /home/guest/b09705011/mnt/data_pair/FTV_selected_seq \
    --valid-percent 0 \
    --dest /home/guest/b09705011/mnt/metadata \
    --output_fname FTV_selected_seq