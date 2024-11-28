# ASCEND + CV16 + NTUCOOL

from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="validation")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/CV16')

dataset = load_dataset("CAiRE/ASCEND", split="validation")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ASCEND')

dataset = load_dataset("ky552/ML2021_ASR_ST", split="dev")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/ML2021_ASR_ST')

dataset = load_dataset("ky552/cszs_zh_en", split="dev")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_eval/cszs_zh_en')

dataset = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/CV16')

dataset = load_dataset("CAiRE/ASCEND", split="test")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ASCEND')

dataset = load_dataset("ky552/ML2021_ASR_ST", split="test")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/ML2021_ASR_ST')

dataset = load_dataset("ky552/cszs_zh_en", split="test")
dataset.save_to_disk('/mnt/home/ntuspeechlabtaipei1/forbes/dataset_test/cszs_zh_en')

