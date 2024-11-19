# python3 filter_data.py \
#     --input /mnt/dataset_1T/BabyBusTC \
#     --max_workers 8

# python3 resample.py \
#     --input /mnt/dataset_1T/FTV_selected \
#     --max_workers 8

python3 resample.py \
    --input /home/guest/b09705011/mnt/dataset_meta \
    --max_workers 4 \
    --invalid_channels ./invalid_channel.tsv


# python3 resample.py \
#     --input /home/guest/b09705011/mnt/makingsashimi \
#     --max_workers 8


# python3 resample.py \
#     --input /mnt/home/ntuspeechlabtaipei1/forbes/dataset_meta \
#     --max_workers 4 \
#     --invalid_channels ./invalid_channel.tsv

# TODO: 看會不會卡 + 有沒有轉到 + 要有 output dir + all_in_one needs minnan_detection

python3 resample.py \
    --input /home/guest/b09705011/mnt/dataset_meta \
    --output_dir /home/guest/b09705011/mnt/_dataset_meta \
    --max_workers 8 \
    --invalid_channels ./invalid_channel.tsv