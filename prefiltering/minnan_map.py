import argparse
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_pinyin(text):
    """提取括號中的拼音"""
    match = re.search(r'（(.+?)）', text)
    return match.group(1) if match else None

def process_chunk(chunk, reference_dict):
    """處理每個數據塊"""
    result = []
    for _, row in chunk.iterrows():
        pinyin = extract_pinyin(row['transcription'])
        if pinyin and pinyin in reference_dict:
            ref_row = reference_dict[pinyin]
            # 保持原始 transcription，更新 idx 和 audio_path
            result.append({
                'idx': ref_row['idx'],
                'transcription': row['transcription'],
                'audio_path': ref_row['audio_path']
            })
        else:
            result.append(row.to_dict())
    return result

def main(file1_path, file2_path, num_workers):
    # 讀取檔案
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep='\t')
    
    # 建立檔案2的拼音對照字典
    reference_dict = {}
    for _, row in df2.iterrows():
        pinyin = extract_pinyin(row['transcription'])
        if pinyin:
            reference_dict[pinyin] = row.to_dict()
    
    # 分割數據為多個塊
    chunk_size = max(1, len(df1) // num_workers)
    chunks = [df1[i:i + chunk_size] for i in range(0, len(df1), chunk_size)]
    
    # 使用多線程處理
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_chunk, chunk, reference_dict)
            for chunk in chunks
        ]
        
        # 使用 tqdm 顯示進度
        for future in tqdm(futures, desc="處理進度"):
            results.extend(future.result())
    
    # 將結果轉換為 DataFrame 並保存
    result_df = pd.DataFrame(results)
    output_path = file1_path.rsplit('.', 1)[0] + '_processed.tsv'
    result_df.to_csv(output_path, sep='\t', index=False)
    print(f"處理完成，結果已保存至：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='檔案對照處理程式')
    parser.add_argument('--file1', required=True, help='需要修改的檔案路徑')
    parser.add_argument('--file2', required=True, help='參考檔案路徑')
    parser.add_argument('--num_workers', type=int, default=4, help='執行緒數量')
    
    args = parser.parse_args()
    main(args.file1, args.file2, args.num_workers)
    
    
    
    
    
    
    
# python3 minnan_map.py --file1 "/mnt/home/ntuspeechlabtaipei1/forbes/CV17手動清洗.tsv" --file2 "/mnt/home/ntuspeechlabtaipei1/forbes/final_dataset/train/CV17_TRAIN_MINNAN/metadata.tsv" --num_workers 180