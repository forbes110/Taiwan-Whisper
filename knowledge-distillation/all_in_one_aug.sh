#!/bin/bash
# huggingface-cli login
# export WANDB_API_KEY=<>

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export NUM_NODES=1   
export GPUS_PER_NODE=8
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

student_model_dir="/mnt/home/ntuspeechlabtaipei1/forbes/student_model"

# 8 * 4 * 8 = 256
per_device_batch_size=4
gradient_accumulation_steps=8

# 16 * 4 * 4
# per_device_batch_size=16
# gradient_accumulation_steps=4

prefetch_factor=64
project_name="aug-4"
max_steps=228000

teacher_model_card="openai/whisper-large-v2"

# train_dataset_manifest=/mnt/home/ntuspeechlabtaipei1/forbes/cleaned_sample/049c00d3-d675-461e-9a11-522694633cdb/cleaned-threshold-0.6-mix_detection.tsv
train_dataset_manifest="/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/train_0.6_20241130_175240.tsv"
eval_dataset_name="mozilla-foundation/common_voice_16_1"
ckpt_dir="/mnt/home/ntuspeechlabtaipei1/forbes/ckpt/$project_name"
preds_dir="/mnt/home/ntuspeechlabtaipei1/forbes/eval_preds"

# if needed
# resume_ckpt="/mnt/home/ntuspeechlabtaipei1/forbes/ckpt/checkpoint-10000-epoch-0"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# python create_student_model.py \
# 	--teacher_checkpoint "$teacher_model_card" \
# 	--encoder_layers 32 \
# 	--decoder_layers 2 \
# 	--save_dir "$student_model_dir" \
# 	--mix_lang_emb

# TODO: Note that resume_from_checkpoint is not always needed
# --resume_from_checkpoint $resume_ckpt \

echo "Distillation start: $(timestamp)" | tee -a run_distillation.log
accelerate launch run_distillation.py \
    --model_name_or_path "$student_model_dir" \
    --teacher_model_name_or_path "$teacher_model_card" \
    --train_dataset_manifest "$train_dataset_manifest" \
    --train_dataset_name "" \
    --train_split_name "" \
    --text_column_name "" \
    --train_dataset_samples "" \
    --eval_dataset_name "$eval_dataset_name" \
    --eval_dataset_config_name "zh-TW" \
    --eval_split_name "validation" \
    --eval_text_column_name "text" \
    --eval_steps 1000 \
    --save_steps 2000 \
    --warmup_steps 50 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "constant_with_warmup" \
    --timestamp_probability 0.5 \
    --condition_on_prev_probability 0.2 \
    --language "zh" \
    --task "transcribe" \
    --logging_steps 100 \
    --save_total_limit 20 \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --ddp_timeout 7200 \
    --dtype "bfloat16" \
    --attn_implementation "flash_attention_2" \
    --output_dir "$ckpt_dir" \
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
    --gradient_accumulation_steps  $gradient_accumulation_steps \
    --dataloader_prefetch_factor $prefetch_factor \
	--mix_lang_emb True \
    --short_form_augmentation True \
    --grain_sec 0.5 \
    --augment_chunk_ratio 0.333 \
    --augment_audio_ratio 0.1 \
    --preds_dir $preds_dir | tee -a run_distillation.log
echo "Distillation complete: $(timestamp)" | tee -a run_distillation.log