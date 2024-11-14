
# TODO: here need to be large-v2
# python3 initial_inference.py \
#     --dataset_path /mnt/dataset_1T/tmp_dir/sample.tsv \
#     --output_dir /mnt/pseudo_label \
#     --language zh \
#     --log_progress True \
#     --model_size tiny \
#     --compute_type default \
#     --chunk_length 5 \
#     --num_workers 8 \
#     --batch_size 16
#!/bin/bash

# python3 initial_inference.py \
#     --dataset_path /mnt/dataset_1T/BabyBusTC.csv \
#     --output_dir /mnt/pseudo_label/BabyBusTC_seq \
#     --language zh \
#     --log_progress True \
#     --model_size tiny \
#     --compute_type default \
#     --chunk_length 5 \
#     --num_workers 8 \
#     --batch_size 16 \
#     --batched_mode 

# python3 initial_inference.py \
#     --dataset_path /mnt/dataset_1T/FTV_selected.tsv \
#     --output_dir /mnt/pseudo_label/FTV_selected \
#     --language zh \
#     --log_progress True \
#     --model_size tiny \
#     --compute_type default \
#     --chunk_length 5 \
#     --num_workers 8 \
#     --batch_size 16 \
#     --batched_mode 

# TODO: note that remove the "_seq" postfix, now is for comparison
python3 initial_inference.py \
    --dataset_path /home/guest/b09705011/mnt/dataset_1T/FTV_selected_.tsv \
    --output_dir /home/guest/b09705011/mnt/pseudo_label/FTV_selected_seq \
    --language zh \
    --log_progress True \
    --model_card large-v2 \
    --compute_type default \
    --chunk_length 5 \
    --num_workers 2 \
    --repetition_penalty 10 \
    --word_timestamps True
    # --batch_size 16 \
    # --batched_mode 
    # --model_card /home/guest/b09705011/mnt/whisper-large-v2-mix-emb \


# python3 initial_inference.py \
#     --dataset_path /home/guest/b09705011/mnt/dataset_1T/FTV_selected.tsv \
#     --output_dir /home/guest/b09705011/mnt/pseudo_label/FTV_selected \
#     --language zh \
#     --log_progress True \
#     --model_card large-v2 \
#     --compute_type default \
#     --chunk_length 5 \
#     --num_workers 2 \
#     --batch_size 16 \
#     --batched_mode 

