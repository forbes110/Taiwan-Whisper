# TODO: streaming == True?
# Do KD
accelerate launch run_distillation.py \
    --model_name_or_path "/mnt/student_model" \
    --teacher_model_name_or_path "openai/whisper-tiny" \
    --train_dataset_manifest "/mnt/cleaned-threshold-0.4.tsv" \
    --train_dataset_name "" \
    --train_split_name "" \
    --text_column_name "" \
    --train_dataset_samples "" \
    --eval_dataset_name "mozilla-foundation/common_voice_16_1" \
    --eval_dataset_config_name "zh-TW" \
    --eval_split_name "validation" \
    --eval_text_column_name "sentence" \
    --eval_steps 1000 \
    --save_steps 5000 \
    --warmup_steps 50 \
    --learning_rate 0.0001 \
    --lr_scheduler_type "constant_with_warmup" \
    --timestamp_probability 0.5 \
    --condition_on_prev_probability 0.2 \
    --language "zh" \
    --task "transcribe" \
    --logging_steps 100 \
    --save_total_limit 20 \
    --max_steps 120000 \
    --wer_threshold 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --ddp_timeout 7200 \
    --dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --output_dir "/mnt/predictions" \
    --do_train \
    --do_eval \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --predict_with_generate \
    --freeze_encoder \
    --freeze_embed_positions \
    --streaming True \
    --is_prefiltered True \
    --skip_audio_length_filtering True \
    --gradient_accumulation_steps 2 \
    --dataloader_prefetch_factor 2 
    # --push_to_hub \ 



# TODO 1.: Note that we now use common voice 16.1 as the temperary eval dataset, may need to change it later, like train/valid split psuedo-labelled.
# TODO 2.: WER or CER threshold?
# TODO 3. check train_dataset_manifest is what? only flac paths or path, text pair?
# TODO 4. mixed_precision