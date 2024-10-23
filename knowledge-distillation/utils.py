import argparse
import opencc
import csv
import editdistance
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from pypinyin import pinyin, lazy_pinyin, Style
from g2p_en import G2p # too slow, should use lexicon instead
import edit_distance
from datasets import IterableDataset, Features
import os
import os.path as osp
import numpy as np
import argparse
import datasets
from copy import deepcopy
from time import sleep
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import soundfile as sf
import re

sampling_rate = 16000
lexicon_fpath = '/knowledge-distillation/lexicon.lst'


def _trim_last_segment(feature: Features):
    timestamp_re_pattern = r"<\|\d{1,2}\.\d{2}\|>"
    timestamps = re.findall(timestamp_re_pattern, feature["whisper_transcript"])
    if len(timestamps) > 1:
        last_timestamp = timestamps[-1]
        # 1. "<|0.00|>...<|29.00|><|29.00|><|continued|>" -> "<|0.00|>...<|29.00|>"
        # 2. "<|0.00|>...<|29.00|>" -> "<|0.00|>...<|29.00|>"
        feature["whisper_transcript"] = feature["whisper_transcript"].split(last_timestamp)[0] + last_timestamp 
        trim_start_frame = int(float(last_timestamp[2:-2]) * sampling_rate)
        if trim_start_frame < len(feature["audio"]["array"]):
            feature["audio"]["array"] = feature["audio"]["array"][:trim_start_frame]
    return feature

def _append_last_segment(feature: Features):
    # re search pattern: <|0.00|> ~ <|30.00|>. re expression: <\|(\d{2}\.\d{2})\|>
    special_tokens_re_pattern = r"<\|[\w\.]{1,12}\|>"
    special_tokens_of_whisper_transcript = re.findall(special_tokens_re_pattern, feature["whisper_transcript"])
    if "<|continued|>" in special_tokens_of_whisper_transcript:
        timestamp_before_continued = special_tokens_of_whisper_transcript[special_tokens_of_whisper_transcript.index("<|continued|>") - 1]
        new_transcript = feature["whisper_transcript"].split(timestamp_before_continued)[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"
        feature["whisper_transcript"] = new_transcript
    else:
        new_transcript = feature["whisper_transcript"].split("<|endoftext|>")[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"

last_segment_handlers = {
    "trim": _trim_last_segment,
    "append": _append_last_segment
}

def cal_complete_mer(ref_data, hyp_data):
    S, D, I, N = (0, 0, 0, 0)
    count = 0
    for ref, hyp in zip(ref_data, hyp_data):
        _S, _D, _I, _N = cal_single_complete_mer(ref, hyp)
        S += _S
        D += _D
        I += _I
        N += _N
        count += 1
    return S, D, I, N, count

def cal_single_complete_mer(ref, hyp):
    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
    opcodes = sm.get_opcodes()
    
    # Substitution
    s = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'replace' else 0) for x in opcodes])
    # Deletion
    d = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'delete' else 0) for x in opcodes])
    # Insertion
    i = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'insert' else 0) for x in opcodes])
    n = len(ref)
    return s, d, i, n

class MixErrorRate(object):
    def __init__(
        self, 
        to_simplified_chinese=True, 
        to_traditional_chinese=False, 
        phonemize=False, 
        separate_language=False, 
        test_only=False,
        count_repetitive_hallucination=False,
        calculate_complete_mer=False
    ):
        self.converter = None
        if to_simplified_chinese and to_traditional_chinese:
            raise ValueError("Can't convert to both simplified and traditional chinese at the same time.")
        if to_simplified_chinese:
            print("Convert to simplified chinese")
            self.converter = opencc.OpenCC('t2s')
        elif to_traditional_chinese:
            print("Convert to traditional chinese")
            self.converter = opencc.OpenCC('s2t')
        else:
            print("No chinese conversion")
        if phonemize:
            if separate_language:
                raise NotImplementedError("Can't separate language and phonemize at the same time.")
            print("Phonemize chinese and english words")
            print("Force traditional to simplified conversion")
            self.converter = opencc.OpenCC('t2s')
            self.zh_phonemizer = partial(lazy_pinyin, style=Style.BOPOMOFO, errors='ignore')
            self.zh_bopomofo_stress_marks = ['ˊ', 'ˇ', 'ˋ', '˙']
            self.en_wrd2phn = defaultdict(lambda: [])
            with open(lexicon_fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    word, phonemes = line.strip().split('\t')
                    self.en_wrd2phn[word] = phonemes.split()
        self.phonemize = phonemize
        self.test_only = test_only
        self.separate_language = separate_language
        self.count_repetitive_hallucination = count_repetitive_hallucination
        self.calculate_complete_mer = calculate_complete_mer
        if self.count_repetitive_hallucination:
            print("Count repetitive hallucination (6gram-5repeat)")
    
    def _from_str_to_list(self, cs_string):
        cs_list = []
        cur_en_word = ''
        for s in cs_string:
            
            # if is space, skip it
            if s in [' ', '\t', '\n', '\r', ',', '.', '!', '?', '。', '，', '！', '？', '、', '；', '：', '「', '」', '『', '』', '（', '）', '(', ')', '\[', '\]', '{', '}', '<', '>', '《', '》', '“', '”', '‘', '’', '…', '—', '～', '·', '•']:
                if cur_en_word != '':
                    cs_list.append(cur_en_word)
                    cur_en_word = ''
                continue
            # if it chinese character, add it to list
            if u'\u4e00' <= s <= u'\u9fff':
                if cur_en_word != '':
                    cs_list.append(cur_en_word)
                    cur_en_word = ''
                if self.converter is not None:
                    s = self.converter.convert(s)
                cs_list.append(s)
            # check character, if it is english character, add it to current word
            elif s.isalnum() or s in ["'", "-"]:
                cur_en_word += s
            else:
                print(f"Unknown character during conversion: {s}")
        if cur_en_word != '':
            cs_list.append(cur_en_word)
        return cs_list
    
    def _unit_is_en(self, token):
        if u'\u4e00' <= token[0] <= u'\u9fff':
            return False
        return True
    
    def _unit_is_zh(self, token):
        if u'\u4e00' <= token[0] <= u'\u9fff':
            return True
        return False

    def _phonemized_cs_list(self, cs_list):
        cur_zh_chars = []
        phonemes = []
        for unit in cs_list:
            if u'\u4e00' <= unit[0] <= u'\u9fff':
                cur_zh_chars.append(unit)
            else:
                if cur_zh_chars:
                    zh_phns = ''.join(self.zh_phonemizer(''.join(cur_zh_chars)))
                    phonemes.extend(filter(lambda p: p not in self.zh_bopomofo_stress_marks, zh_phns))
                    cur_zh_chars = []
                phonemes.extend(self.en_wrd2phn[unit])
        if cur_zh_chars:
            zh_phns = ''.join(self.zh_phonemizer(''.join(cur_zh_chars)))
            phonemes.extend(filter(lambda p: p not in self.zh_bopomofo_stress_marks, zh_phns))
            cur_zh_chars = []
        return phonemes

    def _count_repetitive_hallucination(self, cs_str, n=6, repeat=5, reset_len=100):
        count = 0
        ngram_counts = defaultdict(lambda: 0)
        if len(cs_str) < n:
            return 0
        prev_reset_idx = 0
        for i in range(len(cs_str) - n + 1):
            ngram = cs_str[i:i+n]
            if '|>' in ngram or '<|' in ngram: continue
            ngram_counts[ngram] += 1
            if ngram_counts[ngram] >= repeat:
                count += 1
                # reset for next round calculation
                ngram_counts = defaultdict(lambda: 0)
            if i - prev_reset_idx >= reset_len:
                ngram_counts = defaultdict(lambda: 0)
                prev_reset_idx = i
        return count

    def compute(self, predictions=None, references=None, show_progress=True, empty_error_rate=1.0, **kwargs):
        total_err = 0
        total_ref_len = 0
        total_en_err = 0
        total_en_ref_len = 0
        total_zh_err = 0
        total_zh_ref_len = 0
        repetitive_hallucination_count = 0
        ref_repetitive_hallucination_count = 0
        if self.test_only:
            predictions = predictions[:10]
            references = references[:10]
        if self.calculate_complete_mer:
            S, D, I, N = 0, 0, 0, 0
        iterator = tqdm(enumerate(zip(predictions, references)), total=len(predictions), desc="Computing Mix Error Rate...") if show_progress and len(predictions) > 20 else enumerate(zip(predictions, references))
        for i, (pred, ref) in iterator:
            # if english use word error rate, if chinese use character error rate
            # generate list for editdistance computation
            if self.count_repetitive_hallucination:
                repetitive_hallucination_count += self._count_repetitive_hallucination(pred)
                ref_repetitive_hallucination_count += self._count_repetitive_hallucination(ref)
            pred_list = self._from_str_to_list(pred)
            ref_list = self._from_str_to_list(ref)
            if self.test_only:
                print(f"Prediction List First 20@{i}: {pred_list[:20]}")
                print(f"Reference List First 20@{i}: {ref_list[:20]}")
            if self.phonemize:
                pred_list = self._phonemized_cs_list(pred_list)
                ref_list = self._phonemized_cs_list(ref_list)
            if self.calculate_complete_mer:
                _S, _D, _I, _N = cal_single_complete_mer(ref_list, pred_list)
                S += _S
                D += _D
                I += _I
                N += _N
            # compute edit distance
            if self.separate_language:
                en_pred_list = list(filter(self._unit_is_en, pred_list))
                en_ref_list = list(filter(self._unit_is_en, ref_list))
                zh_pred_list = list(filter(self._unit_is_zh, pred_list))
                zh_ref_list = list(filter(self._unit_is_zh, ref_list))
                en_err = editdistance.eval(en_pred_list, en_ref_list)
                total_en_err += en_err
                total_en_ref_len += len(en_ref_list)
                zh_err = editdistance.eval(zh_pred_list, zh_ref_list)
                total_zh_err += zh_err
                total_zh_ref_len += len(zh_ref_list)
            err = editdistance.eval(pred_list, ref_list)
            total_err += err
            total_ref_len += len(ref_list)
            if self.test_only:
                local_mer = {
                    "MER": err / len(ref_list),
                    "EN WER": en_err / len(en_ref_list) if len(en_ref_list) != 0 else 0,
                    "ZH CER": zh_err / len(zh_ref_list) if len(zh_ref_list) != 0 else 0
                }
                print(f"Local MER@{i}: {local_mer}")
        if total_ref_len == 0:
            print(f"No reference found, return {empty_error_rate*100}% error rate instead")
            return empty_error_rate # if no reference, return 100% error rate instead
        mer = total_err / total_ref_len
        if self.separate_language or self.count_repetitive_hallucination:
            result = {
                "MER": mer,
            }
            if self.separate_language:
                en_wer = total_en_err / total_en_ref_len if total_en_ref_len != 0 else 0
                zh_cer = total_zh_err / total_zh_ref_len if total_zh_ref_len != 0 else 0
                result["EN WER"] = en_wer
                result["ZH CER"] = zh_cer
            if self.count_repetitive_hallucination:
                result["Hyp Repetitive Hallucination Count"] = repetitive_hallucination_count
                result["Ref Repetitive Hallucination Count"] = ref_repetitive_hallucination_count
            return result
        if self.calculate_complete_mer:
            print(f"SUB={S/N}, DEL={D/N}, INS={I/N}, (S, D, I, N)={(S, D, I, N)}, total_len={total_ref_len}")
        return mer # mer
    
    

def load_output_csv(fpath, skip_header=True, hyp_col=1, ref_col=2, delimiter='\t'):
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        predictions = []
        references = []
        if skip_header:
            columns = next(reader)
            print(f"Columns: {columns}")
        for i, row in enumerate(reader):
            predictions.append(row[hyp_col])
            references.append(row[ref_col])
        return predictions, references

def main(args):
    print(args)
    mer = MixErrorRate(
        to_simplified_chinese=args.to_simplified_chinese, 
        to_traditional_chinese=args.to_traditional_chinese, 
        separate_language=args.separate_language,
        test_only=args.test_only,
        count_repetitive_hallucination=args.count_repetitive_hallucination, 
        calculate_complete_mer=args.calculate_complete_mer
    )
    predictions, references = load_output_csv(args.csv_fpath)
    mer_value = mer.compute(predictions=predictions, references=references)
    print(f"Mix Error Rate: {mer_value}")
    
def load_audio_fpaths(manifest_fpath, root=None):
    audio_fpaths = []
    with open(manifest_fpath, "r") as fr:
        if root is None:
            root = fr.readline().strip()
        else:
            _ = fr.readline()
        for line in fr:
            audio_fpath = osp.join(root, line.strip())
            audio_fpaths.append(audio_fpath)
    return audio_fpaths

def customized_data_generator(audio_fpaths, last_segment_handler_type="trim"):
    for audio_fpath in audio_fpaths:
        feature = _get_feature_by_audio_fpath(audio_fpath, last_segment_handler_type=last_segment_handler_type)
        yield feature

def load_customized_dataset(manifest_fpath, root=None) -> IterableDataset:
    print(f"Loading customized dataset from {manifest_fpath}")
    audio_fpaths = load_audio_fpaths(manifest_fpath, root=root)
    ex_feature = Features()
    ex_feature["audio"] = "dummy"
    ex_feature["text"] = "dummy"
    ex_feature['whisper_transcript'] = "dummy"
    ex_feature['last_segment_transcript'] = "dummy"
    ex_feature['condition_on_prev'] = "dummy"
    customized_dataset = IterableDataset.from_generator(customized_data_generator, features=ex_feature, gen_kwargs={"audio_fpaths": audio_fpaths})

    return customized_dataset, audio_fpaths

def _get_feature_by_audio_fpath(audio_fpath, last_segment_handler_type="trim"):
    feature = Features()
    audio_data = sf.read(audio_fpath)[0]
    feature["audio"] = {
        "path": audio_fpath,
        "sampling_rate": sampling_rate,
        "array": audio_data
    }
    with open(audio_fpath.replace(".flac", ".txt"), "r") as trans_fr:
        lines = trans_fr.readlines()
        
        whisper_transcript = lines[0].strip().split("<|endoftext|>")[0] # remove <|endoftext|>
        end_transcript = lines[2].strip()
        prev_transcript = lines[4].strip().split("<|endoftext|>")[0] # remove <|endoftext|>
        
        feature["whisper_transcript"] = whisper_transcript
        feature["last_segment_transcript"] = end_transcript
        feature["condition_on_prev"] = "<|startofprev|>" + prev_transcript
        
        if "<|continued|>" in prev_transcript:
            timestamp_re_pattern = r"<\|\d{1,2}\.\d{2}\|>"
            timestamps = re.findall(timestamp_re_pattern, feature["condition_on_prev"])
            if len(timestamps) > 1:
                last_timestamp = timestamps[-1]
                # 1. "<|0.00|>...<|29.00|><|29.00|><|continued|>" -> "<|0.00|>...<|29.00|>"
                # 2. "<|0.00|>...<|29.00|>" -> "<|0.00|>...<|29.00|>"
                feature["condition_on_prev"] = feature["condition_on_prev"].split(last_timestamp)[0] + last_timestamp
                feature["condition_on_prev"].replace("<|continued|>", "") # ensure that there is no "<|continued|>" in the condition_on_prev
        feature = last_segment_handlers[last_segment_handler_type](feature)
    return feature

def _trim_last_segment(feature: Features):
    timestamp_re_pattern = r"<\|\d{1,2}\.\d{2}\|>"
    timestamps = re.findall(timestamp_re_pattern, feature["whisper_transcript"])
    if len(timestamps) > 1:
        last_timestamp = timestamps[-1]
        # 1. "<|0.00|>...<|29.00|><|29.00|><|continued|>" -> "<|0.00|>...<|29.00|>"
        # 2. "<|0.00|>...<|29.00|>" -> "<|0.00|>...<|29.00|>"
        feature["whisper_transcript"] = feature["whisper_transcript"].split(last_timestamp)[0] + last_timestamp 
        trim_start_frame = int(float(last_timestamp[2:-2]) * sampling_rate)
        if trim_start_frame < len(feature["audio"]["array"]):
            feature["audio"]["array"] = feature["audio"]["array"][:trim_start_frame]
    return feature

def _append_last_segment(feature: Features):
    # re search pattern: <|0.00|> ~ <|30.00|>. re expression: <\|(\d{2}\.\d{2})\|>
    special_tokens_re_pattern = r"<\|[\w\.]{1,12}\|>"
    special_tokens_of_whisper_transcript = re.findall(special_tokens_re_pattern, feature["whisper_transcript"])
    if "<|continued|>" in special_tokens_of_whisper_transcript:
        timestamp_before_continued = special_tokens_of_whisper_transcript[special_tokens_of_whisper_transcript.index("<|continued|>") - 1]
        new_transcript = feature["whisper_transcript"].split(timestamp_before_continued)[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"
        feature["whisper_transcript"] = new_transcript
    else:
        new_transcript = feature["whisper_transcript"].split("<|endoftext|>")[0]
        new_transcript += feature["last_segment_transcript"] + "<|endoftext|>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mix Error Rate")
    parser.add_argument("--csv_fpath", type=str, required=True, help="Path to the csv file")
    parser.add_argument("--to_simplified_chinese", action="store_true", help="Convert chinese to simplified chinese")
    parser.add_argument("--to_traditional_chinese", action="store_true", help="Convert chinese to traditional chinese")
    parser.add_argument("--separate_language", action="store_true", help="Compute MER separately for chinese and english")
    parser.add_argument("--test_only", action="store_true", help="Run test and give some cs_list examples")
    parser.add_argument("--count_repetitive_hallucination", action="store_true", help="Count repetitive hallucination")
    parser.add_argument("--calculate_complete_mer", action="store_true", help="Calculate complete MER")
    args = parser.parse_args()
    main(args)

