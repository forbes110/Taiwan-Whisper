python elim_hallucination.py \
    --original_tsv /mnt/metadata/FTV_selected.tsv \
    --type whisper \
    --hyps_txt /mnt/validator_inference/FTV_selected/validator_inference.txt \
    --output_dir /mnt/cleaned/FTV_selected/ \
    --threshold 0.4 \
    --num_workers 16 \
    --phonemize \
    --mix_detection  