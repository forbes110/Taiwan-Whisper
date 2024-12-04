import pandas as pd

def merge_words_to_segments(word_df, max_duration=4.0):
    """
    將字級標記合併為段落，每段最多4秒
    - 自動處理英文單字間的空格
    - 將「喫」字轉換為「吃」字
    
    參數:
    word_df (pd.DataFrame): 字級標記的DataFrame (需包含 speaker,start,end,text 欄位)
    max_duration (float): 每段最大時長（秒）
    """
    # 處理「喫」字轉換
    word_df['text'] = word_df['text'].str.replace('喫', '吃')
    
    segments = []
    current_segment = {
        'speaker': None,
        'start': None,
        'end': None,
        'text': []
    }
    
    def needs_space(prev_text, curr_text):
        """判斷是否需要在兩個文字之間添加空格"""
        if not prev_text or not curr_text:
            return False
        # 檢查前一個字和當前字是否都是英文字母或數字
        prev_is_eng = any(c.isalpha() and ord(c) < 128 for c in prev_text[-1])
        curr_is_eng = any(c.isalpha() and ord(c) < 128 for c in curr_text[0])
        return prev_is_eng and curr_is_eng
    
    for idx, row in word_df.iterrows():
        current_time = float(row['start'].rstrip('s'))
        
        # 初始化新段落
        if current_segment['start'] is None:
            current_segment['speaker'] = row['speaker']
            current_segment['start'] = row['start']
            current_segment['text'].append(row['text'])
            current_segment['end'] = row['end']
            continue
        
        # 檢查是否需要開始新段落
        segment_start = float(current_segment['start'].rstrip('s'))
        current_duration = current_time - segment_start
        
        if (current_duration > max_duration or 
            row['speaker'] != current_segment['speaker']):
            
            # 完成當前段落
            current_segment['text'] = ''.join(current_segment['text'])
            segments.append(current_segment.copy())
            
            # 開始新段落
            current_segment = {
                'speaker': row['speaker'],
                'start': row['start'],
                'text': [row['text']],
                'end': row['end']
            }
        else:
            # 檢查是否需要添加空格
            if needs_space(current_segment['text'][-1], row['text']):
                current_segment['text'].append(' ')
            # 繼續當前段落
            current_segment['text'].append(row['text'])
            current_segment['end'] = row['end']
    
    # 處理最後一個段落
    if current_segment['text']:
        current_segment['text'] = ''.join(current_segment['text'])
        segments.append(current_segment)
    
    return pd.DataFrame(segments)



# 使用範例
# 讀取字級數據
word_df = pd.read_csv('/mnt/home/ntuspeechlabtaipei1/forbes/pseudo_label_tw_word/AmazingTalker/affyZWvQneg.csv')
segment_df = merge_words_to_segments(word_df)

# 儲存結果
segment_df.to_csv('segment_level.csv', index=False)