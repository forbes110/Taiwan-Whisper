import os
import os.path as osp
import argparse
import glob
import logging
import time
import gc
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import csv

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 30 * SAMPLE_RATE  # (secs * sample_rate = frames)
ADD_CONTINUED_TOKEN_THRESHOLD = 1.0  # (secs)


def frame_diff_to_timestamp(frame_diff, sample_rate=SAMPLE_RATE):
    residual = frame_diff % 320
    
    if 320 - residual > 5 and residual > 5:
        # print(f"Warning: frame_diff {frame_diff} is not very close to a multiple of 320")
        # round frame_diff to the nearest 320 frames
        frame_diff = round(frame_diff / 320) * 320
        
    # if frame_diff is a multiple of 320, then sec_diff is with resolution of 0.02s
    sec_diff = frame_diff / sample_rate 
    
    # use max min function to ensure sec_diff is within [0.00, 30.00]
    sec_diff = max(0.00, min(30.00, sec_diff))
    
    # return token format <|sec_diff:.2f|>
    return f"<|{sec_diff:.2f}|>"

def read_pseudo_labels(csv_fpath):
    """
    pseudo_label takes the form
    start, end, text
    0.25, 18.91, Hello
    """
    segments = []
    with open(csv_fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) != 3:
                continue
            start, end, text = row
            segments.append((float(start), float(end), text.strip()))
    return segments

# Segment audio based on transcriptions
def segment_audio_by_trans(audio_trans_pair, segment_output_dir):
    """
    Segment audio based on transcription.

    Args:
        audio_trans_pair (tuple): (audio_file_path, transcription_file_path)
        segment_output_dir (str): output directory for segmented audio

    Returns:
        str: "Success" on success, otherwise (audio_trans_pair, Exception)
    """
    try:
        audio_fpath, trans_fpath = audio_trans_pair
        
        print(f"Segmenting {audio_fpath} based on {trans_fpath}")   
        
        file_name = osp.basename(audio_fpath).split('.')[0]
        audio_output_dir = osp.join(segment_output_dir, file_name)
        
        os.makedirs(audio_output_dir, exist_ok=True)
        
        audio_data, sr = sf.read(audio_fpath)
        
        """
        segments takes the format
        start,end,text
        0.252,18.391, This is good
        18.391,41.425, Not bad
        41.425,60.967, Oh my god
        60.967,80.862, Nice
        """
        
        segments = read_pseudo_labels(trans_fpath)
        
        
        # TODO: this is added since audio sec count is not always from 0
        prev_end_frame = int(segments[0][0] * SAMPLE_RATE)
        prev_text = ""
        cur_text = ""

        # for each line of a audio transcrion
        for i, segment in enumerate(segments):
            
            start, end, text = segment
            
            print(f"start: {start}, end: {end}, text: {text}")
                        
            s_frame = int(start * SAMPLE_RATE)
            e_frame = int(end * SAMPLE_RATE)
            
            # 4288, 496480
            print(f"s_frame: {s_frame}, e_frame: {end}, text: {e_frame}")
            

            # 0
            s_timetag = frame_diff_to_timestamp(s_frame - prev_end_frame)
            
            # 492192
            e_timetag = frame_diff_to_timestamp(e_frame - prev_end_frame)
            
            # 31.03 - 0.268 
            if e_frame - prev_end_frame > SEGMENT_LENGTH:
                cur_end_frame = prev_end_frame + SEGMENT_LENGTH
                
                segmented_audio = audio_data[prev_end_frame:cur_end_frame]
                
                if cur_end_frame - s_frame > ADD_CONTINUED_TOKEN_THRESHOLD * SAMPLE_RATE:
                    cur_text += s_timetag
                    cur_text += "<|continued|>"
                
                cur_text += "<|endoftext|>"
                
                segment_output_fpath = osp.join(audio_output_dir, f"{file_name}_{prev_end_frame}-{cur_end_frame}.flac")
                sf.write(segment_output_fpath, segmented_audio, SAMPLE_RATE)

                with open(osp.join(audio_output_dir, f"{file_name}_{prev_end_frame}-{cur_end_frame}.txt"), 'w') as f:
                    
                    # current segment
                    f.write(cur_text + "\n")
                    
                    f.write("\n" + s_timetag + text + e_timetag + "\n")
                    
                    # previous segment as promt to the model
                    f.write("\n" + prev_text + "\n")

                prev_end_frame = s_frame
                prev_text = cur_text
                
                s_timetag = frame_diff_to_timestamp(s_frame - prev_end_frame)
                e_timetag  = frame_diff_to_timestamp(e_frame - prev_end_frame)
                cur_text = s_timetag + text + e_timetag                
            else:
                cur_text += s_timetag + text + e_timetag

        return "Success"
    except Exception as e:
        return (audio_trans_pair, e)

# Process all audio files based on transcriptions
def segment_audio(audio_dir, trans_dir, segment_output_dir):
    
    # Build a mapping from video IDs to transcription files
    pseudo_label_fpath = {}
    
    # all pseudo label path generated by whisper-large
    trans_fpaths = list(glob.glob(osp.join(trans_dir, '*.csv'), recursive=True))

    for trans_fpath in tqdm(trans_fpaths, desc="Parsing transcriptions..."):
        file_name = osp.basename(trans_fpath).split('.')[0]
        pseudo_label_fpath[file_name] = trans_fpath
        
    # Find and pair all audio files with their transcriptions
    audio_trans_pairs = []
    audio_fpaths = glob.glob(osp.join(audio_dir, '*.flac'))

    for audio_fpath in audio_fpaths:
        file_name = osp.basename(audio_fpath).split('.')[0]
        trans_fpath = pseudo_label_fpath.get(file_name)
        
        if trans_fpath is None:
            print(f"Warning: No transcription found for {file_name}")
            continue
        
        # audio path, pseudo_label path pair
        audio_trans_pairs.append((audio_fpath, trans_fpath))

    # Process audio files in parallel using multiprocessing
    chunk_size = 100
    
    segment_func = partial(segment_audio_by_trans, segment_output_dir=segment_output_dir)
    
    for i in range(0, len(audio_trans_pairs), chunk_size):
        # print(f"audio_trans_pairs: {audio_trans_pairs}")
        
        end_i = min(i + chunk_size, len(audio_trans_pairs))
        # chunk of audio piars to accelerate by parallel processing
        chunk = audio_trans_pairs[i:end_i]
        
        print(f"Processing chunk {i}-{i + end_i} with {args.nprocs} processes...")

        with mp.Pool(processes=args.nprocs) as pool:
            for result in tqdm(pool.imap_unordered(segment_func, chunk), total=len(chunk), desc="Segmenting audio..."):
                if result != "Success":
                    print(f"Error: Failed to segment {result[0]}, error={result[1]}", flush=True)
        gc.collect()
    print("Done")

def main(args):
    print(args)
    segment_audio(args.audio_dir, args.trans_dir, args.segment_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_dir", required=True, help="Root directory of transcriptions")
    parser.add_argument("--segment_output_dir", required=True, help="segment output directory")
    parser.add_argument("--audio_dir", required=True, help="audio directory")
    parser.add_argument("--nprocs", type=int, default=8, help="Number of processes for parallel segmentation")
    args = parser.parse_args()

    main(args)