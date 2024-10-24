dataset_name="mozilla-foundation/common_voice_16_1"
dataset_config_name="zh-TW"
dataset_split_name="test"
text_column_name="sentence"
audio_column_name="audio"

python run_eval.py \
    --model_name_or_path openai/whisper-tiny \
    --dataset_name $dataset_name \
    --dataset_config_name $dataset_config_name \
    --dataset_split_name $dataset_split_name \
    --text_column_name $text_column_name \
    --audio_column_name $audio_column_name \
    --batch_size 32 \
    --dtype "bfloat16" \
    --generation_max_length 256 \
    --language "zh" \
    --attn_implementation "sdpa" \
    --mix_lang_emb True 

