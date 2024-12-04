import argparse
import pandas as pd
import soundfile as sf
import os
import math
import librosa
import numpy as np

def round_to_nearest_002(number):
    """將數字四捨五入到最接近的 0.02 的倍數，且確保不小於 0.02"""
    rounded = round(number / 0.02) * 0.02
    return max(0.02, rounded)

def resample_and_save(input_path, output_path, target_sr=16000):
    """重採樣音頻文件並保存"""
    # 讀取音頻
    audio, sr = librosa.load(input_path, sr=None)
    
    # 如果採樣率不是16000Hz，進行重採樣
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存為 FLAC 格式
    sf.write(output_path, audio, target_sr, format='FLAC')
    
    return audio, target_sr

def process_file(file_name):
    """處理 TSV 文件，重採樣音頻並生成對應的文本文件"""
    # 讀取 TSV 文件
    df = pd.read_csv(file_name, sep='\t')
    
    for _, row in df.iterrows():
        idx = row['idx']
        transcription = row['transcription']
        audio_path = row['audio_path']
        
        # 構建重採樣後的音頻輸出路徑
        output_dir = os.path.dirname(audio_path)
        resampled_audio_path = audio_path  # 直接覆蓋原檔案
        
        # 重採樣音頻文件
        audio, sr = resample_and_save(audio_path, resampled_audio_path)
        
        # 計算新的音頻時長
        duration = len(audio) / sr
        rounded_duration = round_to_nearest_002(duration)
        
        # 構建輸出文本
        output_text = f"<|0.02|>{transcription} <|{rounded_duration:.2f}|><|endfortext|>"
        
        # 構建文本輸出路徑
        output_file = os.path.join(output_dir, f"{idx}.txt")
        
        # 寫入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"已處理：{idx}")
        print(f"  - 重採樣至 16000Hz")
        print(f"  - 生成文本文件：{output_file}")

def main():
    parser = argparse.ArgumentParser(description='處理音頻元數據，重採樣音頻並生成文本文件')
    parser.add_argument('--file_name', required=True, help='TSV 文件的路徑')
    args = parser.parse_args()
    
    process_file(args.file_name)

if __name__ == "__main__":
    main()
    
    
    
"""
python3 /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/process_labelled_train_data.py --file_name /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/ASCEND_TRAIN/metadata.tsv
python3 /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/process_labelled_train_data.py --file_name /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV16_OTHER/metadata.tsv
python3 /mnt/home/ntuspeechlabtaipei1/forbes/Taiwan-Whisper/prefiltering/process_labelled_train_data.py --file_name /mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV16_TRAIN/metadata.tsv
"""