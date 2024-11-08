python3 validator_inference.py \
    --manifest /mnt/metadata/FTV_selected.tsv \
    --output_dir /mnt/validator_inference/FTV_selected \
    --validator openai/whisper-tiny \
    --batch_size 64

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