import re
import unicodedata
from evaluation import MixErrorRate

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class CodeSwitchNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        # 基本清理
        s = s.lower()
        s = re.sub(r'<\|[0-9.]+\|>', ' ', s)  
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        
        # 符號清理
        s = self.clean(s).lower()
        
        # 處理空格
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        
        # 處理中英文混合的情況
        s = self._handle_code_switching(s)
            
        return s

    def _handle_code_switching(self, text):
        # 先將文字分割成詞組
        words = text.split()
        
        # 合併連續的中文字
        result = []
        current_chinese = []
        
        for word in words:
            # 檢查是否為中文字
            if all('\u4e00' <= char <= '\u9fff' for char in word):
                current_chinese.append(word)
            else:
                # 如果有累積的中文字，先處理它們
                if current_chinese:
                    result.append(''.join(current_chinese))
                    current_chinese = []
                result.append(word)
        
        # 處理最後可能剩餘的中文字
        if current_chinese:
            result.append(''.join(current_chinese))
        
        return ' '.join(result)

# 測試程式碼
if __name__ == "__main__":
    normalizer = CodeSwitchNormalizer()
    metric = MixErrorRate()
    
    
    # 測試案例
    test_strings = [
        "<|0.00|>with<|0.36|><|0.36|>your<|1.00|><|1.00|>and<|1.80|><|1.80|>your<|2.40|><|2.40|>hobbies?<|2.84|>",
         "<|0.00|>因為<|1.00|><|1.00|>感覺<|1.54|><|1.54|>香港<|2.00|><|2.00|>好<|2.20|><|2.20|>多<|2.32|><|2.32|>都<|2.48|><|2.48|>深<|2.64|><|2.64|>鎮<|2.80|><|2.80|>人<|3.08|>",
        "<|0.00|>Today<|0.10|><|0.10|>我想去買<|0.90|><|0.90|>coffee<|1.00|>",
        "<|0.00|>我喜歡drinking a lot of 咖啡<|1.00|>",
        "<|0.00|>我喜歡drinking<|0.20|><|0.20|>a<|0.50|><|0.50|>lot<|0.70|><|0.70|>of 咖啡<|1.00|>",
        "<|0.00|>那<|0.28|><|0.28|>我<|0.50|><|0.50|>居<|0.64|><|0.64|>然<|0.74|><|0.74|>是<|1.00|><|1.00|>那個<|1.36|><|1.36|>IS<|1.70|><|1.70|>M<|2.00|><|2.00|>information<|2.48|><|2.48|>system<|3.00|><|3.00|>management<|3.44|><|3.44|>Management<|3.80|>"
    ]
    
    for test_string in test_strings:
        normalized_text = normalizer(test_string)
        print("正規化前:", test_string)
        print("正規化後:", normalized_text)

# 測試程式碼
# if __name__ == "__main__":
#     metric = MixErrorRate()

#     def compute_metrics(pred_str, label_str):
#         # normalize everything and re-compute the mer
#         norm_pred_str = [normalizer(pred) for pred in pred_str]
#         norm_label_str = [normalizer(label) for label in label_str]

#         # filtering step to only evaluate the samples that correspond to non-zero normalized references:
#         norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
#         norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

#         mer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
#         return mer
    
#     normalizer = CodeSwitchNormalizer()
#     test_string = "<|0.00|>with<|0.36|><|0.36|>your<|1.00|><|1.00|>and<|1.80|><|1.80|>your<|2.40|><|2.40|>hobbies?<|2.84|>"
#     test_string = "<|0.00|>因為<|1.00|><|1.00|>感覺<|1.54|><|1.54|>香港<|2.00|><|2.00|>好<|2.20|><|2.20|>多<|2.32|><|2.32|>都<|2.48|><|2.48|>深<|2.64|><|2.64|>鎮<|2.80|><|2.80|>人<|3.08|>"
#     test_string = "<|0.00|>吃<|0.46|><|0.46|>葡<|0.68|><|0.68|>萄<|0.76|><|0.76|>不<|0.92|><|0.92|>讀<|1.08|><|1.08|>葡<|1.24|><|1.24|>萄<|1.36|><|1.36|>瓶<|1.64|><|1.64|>冰<|1.76|>"
#     ref = "吃葡萄不讀葡萄瓶冰"
    
#     pred = normalizer(test_string)
    
    
#     print("正規化前:", test_string)
#     print("正規化後(pred):", pred)
#     print("ref:", ref)
    
#     mer = compute_metrics([pred], [ref])
#     print("mer:", mer)
    