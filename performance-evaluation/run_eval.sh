#!/bin/bash
dataset_name="mozilla-foundation/common_voice_16_1"
dataset_config_name="zh-TW"
dataset_split_name="test"
text_column_name="sentence"
audio_column_name="audio"
save_dir="/mnt/home/ntuspeechlabtaipei1/forbes/test_prediction"


# dataset_name="CAiRE/ASCEND"
# dataset_config_name="main"
# dataset_split_name="test"
# text_column_name="transcription"
# audio_column_name="audio"
save_dir="/mnt/home/ntuspeechlabtaipei1/forbes/test_prediction_cv16_whisper"
# --model_name_or_path /mnt/home/ntuspeechlabtaipei1/forbes/ckpt/normal-8-best-checkpoint-epoch-1 \
# model_name_or_path=/mnt/home/ntuspeechlabtaipei1/forbes/ckpt/normal-8
model_name_or_path=openai/whisper-large-v2

# python run_eval.py \
#     --model_name_or_path openai/whisper-large-v2 \
#     --dataset_name "$dataset_name" \
#     --dataset_config_name "$dataset_config_name" \
#     --dataset_split_name "$dataset_split_name" \
#     --text_column_name "$text_column_name" \
#     --audio_column_name "$audio_column_name" \
#     --batch_size 32 \
#     --dtype "bfloat16" \
#     --generation_max_length 512 \
#     --language "zh" \
#     --attn_implementation "flash_attention_2" \
#     --mix_lang_emb True \
#     --save_dir "$save_dir"


python run_eval.py \
    --model_name_or_path $model_name_or_path \
    --dataset_name "$dataset_name" \
    --dataset_config_name "$dataset_config_name" \
    --dataset_split_name "$dataset_split_name" \
    --text_column_name "$text_column_name" \
    --audio_column_name "$audio_column_name" \
    --batch_size 32 \
    --dtype "bfloat16" \
    --generation_max_length 512 \
    --language "zh" \
    --attn_implementation "flash_attention_2" \
    --mix_lang_emb True \
    --save_dir "$save_dir"
