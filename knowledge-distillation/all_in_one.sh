#!/bin/bash
# huggingface-cli login

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8  
export NUM_NODES=1   
export GPUS_PER_NODE=8  
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

student_model_dir="/mnt/home/ntuspeechlabtaipei1/forbes/student_model"

# TODO: just test
# teacher_model_card="openai/whisper-large-v2"
teacher_model_card="openai/whisper-tiny"
# train_dataset_manifest="/mnt/home/ntuspeechlabtaipei1/forbes/cleaned-threshold-0.6.tsv"
train_dataset_manifest="/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/train_0.6.tsv"
eval_dataset_name="mozilla-foundation/common_voice_16_1"
eval_pred_dir="/mnt/home/ntuspeechlabtaipei1/forbes/eval_preds"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# for check

python create_student_model.py \
	--teacher_checkpoint "$teacher_model_card" \
	--encoder_layers 2 \
	--decoder_layers 1 \
	--save_dir "$student_model_dir" \
	--mix_lang_emb

# python create_student_model.py \
# 	--teacher_checkpoint "$teacher_model_card" \
# 	--encoder_layers 32 \
# 	--decoder_layers 2 \
# 	--save_dir "$student_model_dir" \
# 	--mix_lang_emb

#TODO: note that the dataset {train, validation} both need to be prepared first
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
    --eval_text_column_name "sentence" \
    --eval_steps 1000 \
    --save_steps 5000 \
    --warmup_steps 50 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "constant_with_warmup" \
    --timestamp_probability 0.5 \
    --condition_on_prev_probability 0.2 \
    --language "zh" \
    --task "transcribe" \
    --logging_steps 100 \
    --save_total_limit 20 \
    --max_steps 120000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --ddp_timeout 7200 \
    --dtype "bfloat16" \
    --attn_implementation "sdpa" \
    --output_dir "$eval_pred_dir" \
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
    --dataloader_prefetch_factor 2 \
	--mix_lang_emb True
echo "Distillation complete: $(timestamp)" | tee -a run_distillation.log


    # --attn_implementation "flash_attention_2" \