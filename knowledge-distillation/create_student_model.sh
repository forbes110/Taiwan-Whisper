# init
# python create_student_model.py \
#   --teacher_checkpoint "openai/whisper-large-v2" \
#   --encoder_layers 32 \
#   --decoder_layers 2 \
#   --save_dir "/mnt/student_model"\
#   --mix_lang_emb

# for check
python create_student_model.py \
  --teacher_checkpoint "openai/whisper-tiny" \
  --encoder_layers 1 \
  --decoder_layers 2 \
  --save_dir "/mnt/student_model"\
  --mix_lang_emb

