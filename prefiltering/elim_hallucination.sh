python elim_hallucination.py \
    --original_tsv /home/guest/b09705011/mnt/metadata/FTV_selected_seq.tsv \
    --type whisper \
    --hyps_txt /home/guest/b09705011/mnt/validator_inference/FTV_selected_seq/validator_inference.txt \
    --output_dir /home/guest/b09705011/mnt/cleaned/FTV_selected_seq/ \
    --threshold 0.4 \
    --num_workers 16 \
    --phonemize \
    --mix_detection  