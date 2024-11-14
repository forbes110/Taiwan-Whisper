python3 validator_inference.py \
    --manifest /home/guest/b09705011/mnt/metadata/FTV_selected_seq.tsv \
    --output_dir /home/guest/b09705011/mnt/validator_inference/FTV_selected_seq \
    --validator openai/whisper-medium \
    --batch_size 32

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--root",
#     default=None,
#     help="a sample arg",
# )
# parser.add_argument(
#     "--manifest",
#     default="",
#     required=True,
#     help="a sample arg",
# )
# args = parser.parse_args()

# main(args)