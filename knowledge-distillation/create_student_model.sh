# init
# python create_student_model.py \
#   --teacher_checkpoint "openai/whisper-large-v2" \
#   --encoder_layers 32 \
#   --decoder_layers 2 \
#   --save_dir "/mnt/student_model"\
#   --mix_lang_emb

# for check
# TODO: here need to be modified, only for test
python create_student_model.py \
  --teacher_checkpoint "openai/whisper-tiny" \
  --encoder_layers 1 \
  --decoder_layers 2 \
  --save_dir "/home/guest/b09705011/mnt/student_model"\
  --mix_lang_emb

