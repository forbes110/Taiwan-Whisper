python elim_hallucination.py \
    --original_tsv /home/guest/b09705011/mnt/metadata/FTV_selected_seq.tsv \
    --type whisper \
    --hyps_txt /home/guest/b09705011/mnt/validator_inference/FTV_selected_seq/validator_inference.txt \
    --output_dir /home/guest/b09705011/mnt/cleaned/FTV_selected_seq/ \
    --threshold 0.4 \
    --num_workers 16 \
    --phonemize \
    --mix_detection  



python elim_hallucination.py \
    --original_tsv /mnt/home/ntuspeechlabtaipei1/forbes/metadata/Awater.tsv \
    --type whisper \
    --hyps_txt /mnt/home/ntuspeechlabtaipei1/forbes/validator_inference/Awater/validator_inference.txt \
    --output_dir /mnt/home/ntuspeechlabtaipei1/forbes/cleaned/Awater/ \
    --threshold 0.5 \
    --num_workers 16 \
    --phonemize \
    --mix_detection  