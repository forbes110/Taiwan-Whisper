import os
import os.path as osp
import argparse
import glob
import logging
import time
import gc
import librosa
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 30 * SAMPLE_RATE  # 30 seconds * 16000 Hz = 480000 frames
ADD_CONTINUED_TOKEN_THRESHOLD = 1.0  # (seconds)

def frame_diff_to_timestamp(frame_diff, sample_rate=SAMPLE_RATE):
    residual = frame_diff % 320
    
    if 320 - residual > 5 and residual > 5:
        frame_diff = round(frame_diff / 320) * 320
    
    sec_diff = frame_diff / sample_rate
    sec_diff = max(0.00, min(30.00, sec_diff))  # Ensure sec_diff is within [0.00, 30.00]
    return f"<|{sec_diff:.2f}|>"

def read_pseudo_labels(csv_fpath):
    """Read pseudo-labels from CSV."""
    segments = []
    with open(csv_fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            
            if len(row) == 4:
                speaker, start, end, text = row
            elif len(row) == 3:
                start, end, text = row
            else:
                continue
            segments.append((float(start.rstrip('s')), float(end.rstrip('s')), text.strip()))
    return segments

def segment_audio_by_trans(audio_trans_pair, segment_output_dir):
    """
    Segment audio using transcription data.

    Args:
        audio_trans_pair (tuple): (audio_file_path, transcription_file_path)
        segment_output_dir (str): Output directory for segmented audio.
    """
    try:
        audio_fpath, trans_fpath = audio_trans_pair
        print(f"Segmenting {audio_fpath} based on {trans_fpath}")
        file_name = osp.basename(audio_fpath).split('.')[0]
        audio_output_dir = osp.join(segment_output_dir, file_name)
        os.makedirs(audio_output_dir, exist_ok=True)

        # Load audio using librosa
        audio_data, sr = librosa.load(audio_fpath, sr=SAMPLE_RATE)

        segments = read_pseudo_labels(trans_fpath)

        prev_end_frame = int(segments[0][0] * SAMPLE_RATE)
        prev_e_timetag = "<|0.00|>"
        prev_seg_end_frame = int(segments[0][0] * SAMPLE_RATE)
        prev_text = ""
        cur_text = ""

        """
        segments takes the format
        start,end,text
        0.252,18.391, This is good
        18.391,41.425, Not bad
        41.425,60.967, Oh my god
        60.967,80.862, Nice
        """
        for i, (start, end, text) in enumerate(segments):
            s_frame = int(start * SAMPLE_RATE)
            e_frame = int(end * SAMPLE_RATE)
            
            if s_frame != prev_seg_end_frame:
                cur_text += prev_e_timetag

            s_timetag = frame_diff_to_timestamp(s_frame - prev_end_frame)
            e_timetag = frame_diff_to_timestamp(e_frame - prev_end_frame)
            
            # save 30 sec
            if e_frame - prev_end_frame > SEGMENT_LENGTH:
                cur_end_frame = prev_end_frame + SEGMENT_LENGTH
                
                if prev_end_frame != prev_seg_end_frame:

                    # Segment audio within the allowed range
                    segmented_audio = audio_data[prev_end_frame:prev_seg_end_frame]

                    if cur_end_frame - s_frame > ADD_CONTINUED_TOKEN_THRESHOLD * SAMPLE_RATE:
                        cur_text += s_timetag + "<|continued|>"

                    cur_text += "<|endoftext|>"

                    segment_output_fpath = osp.join(
                        audio_output_dir, f"{file_name}_{prev_end_frame}-{prev_seg_end_frame}.flac"
                    )

                    # Save audio segment using soundfile
                    sf.write(segment_output_fpath, segmented_audio, SAMPLE_RATE)

                    # Save transcription
                    with open(osp.join(audio_output_dir, f"{file_name}_{prev_end_frame}-{prev_seg_end_frame}.txt"), 'w') as f:
                        f.write(cur_text + "\n")
                        f.write(f"\n{s_timetag}{text}{e_timetag}\n")
                        f.write(f"\n{prev_text}\n")
                    
                # NEXT FILE

                """
                Handle the case of no speech but with background music, e.g.,
                Need to consider this case:
                28.228, 30.455, 誒! (but prev_end_frame is calculated 35.711)
                35.711, 37.884, 你好
                ...
                57.795, 60.256, 嘿嘿嘿嘿   
                """
                prev_end_frame = prev_seg_end_frame
                   
                s_timetag = frame_diff_to_timestamp(s_frame - prev_end_frame)
                e_timetag = frame_diff_to_timestamp(e_frame - prev_end_frame)
                                
                prev_e_timetag = e_timetag
                prev_text = cur_text
                
                if s_frame != prev_seg_end_frame:
                    cur_text = "<|0.00|>" + s_timetag + s_timetag + text + e_timetag
                else:
                    cur_text = "<|0.00|>" + text + e_timetag
                    
            # else
            else:
                if s_frame != prev_seg_end_frame:
                    cur_text += s_timetag
                cur_text += s_timetag + text + e_timetag
            
            # handle the case of no speech but with background music
            prev_seg_end_frame = e_frame
            prev_e_timetag = e_timetag
        return "Success"
    except Exception as e:
        return (audio_trans_pair, e)

def segment_audio(audio_dir, trans_dir, segment_output_dir):
    """Process all audio files based on their transcriptions."""
    pseudo_label_fpath = {}

    # print("trans_dir", trans_dir)
    trans_fpaths = glob.glob(osp.join(trans_dir, '*.csv'))
    # print("==========================trans_fpaths", trans_fpaths)
    
    for trans_fpath in tqdm(trans_fpaths, desc="Parsing transcriptions..."):
        file_name = osp.basename(trans_fpath).split('.')[0]
        pseudo_label_fpath[file_name] = trans_fpath

    audio_trans_pairs = []
    
    # Note that the file saved would be flac
    # print("trans_fpaths:", trans_fpaths)
    # print("audio_dir:", audio_dir) 
    audio_fpaths = glob.glob(osp.join(audio_dir, '*.m4a')) + glob.glob(osp.join(audio_dir, '*.flac'))
    # print("audio_fpaths:", audio_fpaths)

    for audio_fpath in audio_fpaths:
        file_name = osp.basename(audio_fpath).split('.')[0]
        trans_fpath = pseudo_label_fpath.get(file_name)

        if trans_fpath is None:
            print(f"Warning: No transcription found for {file_name}")
            continue

        audio_trans_pairs.append((audio_fpath, trans_fpath))

    chunk_size = 100
    os.makedirs(segment_output_dir, exist_ok=True)
    segment_func = partial(segment_audio_by_trans, segment_output_dir=segment_output_dir)

    for i in range(0, len(audio_trans_pairs), chunk_size):
        chunk = audio_trans_pairs[i:i + chunk_size]
        print(f"Processing chunk {i}-{i + len(chunk)} with {args.nprocs} processes...")

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
    parser.add_argument("--segment_output_dir", required=True, help="Segment output directory")
    parser.add_argument("--audio_dir", required=True, help="Audio directory")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of processes for parallel segmentation")
    args = parser.parse_args()

    main(args)
