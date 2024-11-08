# from transformers.models.whisper.english_normalizer import BasicTextNormalizer
# import argparse
# import os.path as osp
# import os
# import re
# import pandas as pd

# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(description="Remove common hallucination from transcriptions.")
#     parser.add_argument(
#         "--original_tsv",
#         type=str,
#         default="/mnt/audio_paths.tsv",
#         help="Path to TSV file containing audio file paths."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/mnt/common_hallucination_caught",
#         help="Directory to save the results."
#     )
#     parser.add_argument(
#         "--execute_removal",
#         action='store_true',
#         help="Flag to execute the removal of transcription files containing hallucinations."
#     )
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     output_dir = args.output_dir
#     if not osp.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")
    
#     hallucination_match_list = [
#         "Okay.",
#         "...",
#         ".",
#         "Mm.",
#     ]

#     hallucination_contain_list = [
#         "請不吝",
#         r"(?<!\w)org(?!\w)",  # ensure examples like "organization" are not removed
#         "點贊",
#         "點讚",
#         "字幕提供",
#         "支持明鏡",
#         "點點欄目"
#     ]
#     # Initialize the lists to store results
#     contain_results = []
#     match_results = []

#     normalizer = BasicTextNormalizer()

#     # Load original TSV for audio paths
#     with open(args.original_tsv, "r", encoding="utf-8") as f:
#         root = f.readline().strip()
#         audio_subfpaths = [l.strip() for l in f.readlines()]
#         audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
#         trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]

#     for trans_fpath in trans_fpaths:
#         if not osp.exists(trans_fpath):
#             print(f"Transcription file not found: {trans_fpath}")
#             continue  # Skip if transcription file does not exist

#         with open(trans_fpath, "r", encoding="utf-8") as f:
#             lines = f.readlines()
            
#             if not lines:
#                 print(f"Empty transcription file: {trans_fpath}")
#                 continue  # Skip empty files
            
#             # Remove <|endoftext|>
#             whisper_transcript = lines[0].strip().split("<|endoftext|>")[0]
            
#             # Remove <|continued|>
#             whisper_transcript = whisper_transcript.split("<|continued|>")[0] 
            
#             # Previous segment as prompt (optional, based on your data structure)
#             if len(lines) > 1:
#                 end_transcript = lines[1].strip()
#             else:
#                 end_transcript = ""
            
#             # Find all timestamp tokens ('<|*|>') and remove them in the transcript
#             timestamp_tokens = re.findall(r"<\|\d{1,2}\.\d{2}\|>", whisper_transcript + end_transcript)
            
#             for st in timestamp_tokens:
#                 whisper_transcript = whisper_transcript.replace(st, ' ')
#             whisper_transcript = whisper_transcript.strip().replace('  ', ' ')
            
#             # Standardize the transcript
#             whisper_transcript = normalizer(whisper_transcript)
                        
#             # Check for contains cases
#             matched_keywords = [keyword for keyword in hallucination_contain_list if re.search(keyword, whisper_transcript)]
#             if matched_keywords:
#                 contain_results.append({
#                     "trans_fpath": trans_fpath,
#                     "matched_keywords": ', '.join(matched_keywords),
#                     "transcript": whisper_transcript
#                 })

#             # Check for match cases
#             words = re.findall(r'\b\w+\b|\.\.\.|[^\s\w]', whisper_transcript)
#             matched_words = [word for word in words if word in hallucination_match_list]
#             if matched_words:
#                 match_results.append({
#                     "trans_fpath": trans_fpath,
#                     "matched_words": ', '.join(matched_words),
#                     "transcript": whisper_transcript
#                 })

#     # Save results to CSV with UTF-8 encoding
#     if contain_results:
#         contain_df = pd.DataFrame(contain_results)
#         contain_df.to_csv(osp.join(output_dir, "contain_cases.csv"), index=False, encoding="utf-8")
#         print(f"Saved contain cases to {osp.join(output_dir, 'contain_cases.csv')}")
#     if match_results:
#         match_df = pd.DataFrame(match_results)
#         match_df.to_csv(osp.join(output_dir, "match_cases.csv"), index=False, encoding="utf-8")
#         print(f"Saved match cases to {osp.join(output_dir, 'match_cases.csv')}")
    
#     if not match_results and not contain_results:
#         print("No common hallucination found.")
    
#     if args.execute_removal:
#         # Collect all trans_fpaths that need to be removed
#         trans_fpaths_to_remove = set()
#         for result in contain_results + match_results:
#             trans_fpaths_to_remove.add(result["trans_fpath"])

#         if not trans_fpaths_to_remove:
#             print("No transcription files to remove based on the identified hallucinations.")
#             return

#         print(f"Total transcription files to remove: {len(trans_fpaths_to_remove)}")

#         for trans_fpath in trans_fpaths_to_remove:
#             if osp.exists(trans_fpath):
#                 try:
#                     os.remove(trans_fpath)
#                     print(f"Removed transcription file: {trans_fpath}")
#                 except Exception as e:
#                     print(f"Error removing transcription file {trans_fpath}: {e}")
#             else:
#                 print(f"Transcription file not found (already removed?): {trans_fpath}")
        
#         # Optionally, update the original TSV if necessary
#         # Since audio files are not removed, the TSV may not need updating.
#         # If the TSV only lists audio files, and you want to keep track of which have missing transcriptions,
#         # you might consider creating a separate log or updating accordingly.
#         # Below is an example of creating an updated TSV excluding audio files whose transcriptions were removed.

#         # Create a set of audio files to keep (those whose transcriptions are not removed)
#         audio_fpaths_to_keep = set(audio_fpaths) - {trans_fpath.replace('txt', 'flac') for trans_fpath in trans_fpaths_to_remove}

#         # Note: If you don't want to modify the original TSV, you can skip this part.
#         # If you do want to create an updated TSV:
#         updated_tsv_path = osp.join(output_dir, "updated_audio_paths.tsv")
#         with open(updated_tsv_path, "w", encoding="utf-8") as f:
#             f.write(root + "\n")
#             for audio_fpath in audio_fpaths_to_keep:
#                 relative_path = osp.relpath(audio_fpath, root)
#                 f.write(f"{relative_path}\n")
#         print(f"Updated TSV saved to: {updated_tsv_path}")

# if __name__ == "__main__":
#     main()

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse
import os.path as osp
import os
import re
import pandas as pd

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Remove common hallucinations from transcriptions.")
    parser.add_argument(
        "--original_tsv",
        type=str,
        default="/mnt/audio_paths.tsv",
        help="Path to TSV file containing audio file paths."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/common_hallucination_caught/FTV_selected",
        help="Directory to save the results."
    )
    parser.add_argument(
        "--execute_removal",
        action='store_true',
        help="Flag to remove paths from the TSV file if they contain hallucinations, without deleting audio files."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    hallucination_match_list = [
        "Okay.",
        "...",
        ".",
        "Mm.",
    ]

    hallucination_contain_list = [
        "請不吝",
        r"(?<!\w)org(?!\w)",  # Ensure examples like "organization" are not removed
        "點贊",
        "點讚",
        "字幕提供",
        "支持明鏡",
        "點點欄目"
    ]
    # Initialize lists to store results
    contain_results = []
    match_results = []

    normalizer = BasicTextNormalizer()

    # Load original TSV for audio paths
    with open(args.original_tsv, "r", encoding="utf-8") as f:
        root = f.readline().strip()
        audio_subfpaths = [l.strip() for l in f.readlines()]
        audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
        trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]

    for trans_fpath in trans_fpaths:
        if not osp.exists(trans_fpath):
            print(f"Transcription file not found: {trans_fpath}")
            continue  # Skip if transcription file does not exist

        with open(trans_fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            if not lines:
                print(f"Empty transcription file: {trans_fpath}")
                continue  # Skip empty files
            
            # Remove <|endoftext|>
            whisper_transcript = lines[0].strip().split("<|endoftext|>")[0]
            
            # Remove <|continued|>
            whisper_transcript = whisper_transcript.split("<|continued|>")[0] 
            
            # Previous segment as prompt (optional, based on your data structure)
            if len(lines) > 1:
                end_transcript = lines[1].strip()
            else:
                end_transcript = ""
            
            # Find all timestamp tokens ('<|*|>') and remove them in the transcript
            timestamp_tokens = re.findall(r"<\|\d{1,2}\.\d{2}\|>", whisper_transcript + end_transcript)
            
            for st in timestamp_tokens:
                whisper_transcript = whisper_transcript.replace(st, ' ')
            whisper_transcript = whisper_transcript.strip().replace('  ', ' ')
            
            # Standardize the transcript
            whisper_transcript = normalizer(whisper_transcript)
                        
            # Check for contains cases
            matched_keywords = [keyword for keyword in hallucination_contain_list if re.search(keyword, whisper_transcript)]
            if matched_keywords:
                contain_results.append({
                    "trans_fpath": trans_fpath,
                    "matched_keywords": ', '.join(matched_keywords),
                    "transcript": whisper_transcript
                })

            # Check for match cases
            words = re.findall(r'\b\w+\b|\.\.\.|[^\s\w]', whisper_transcript)
            matched_words = [word for word in words if word in hallucination_match_list]
            if matched_words:
                match_results.append({
                    "trans_fpath": trans_fpath,
                    "matched_words": ', '.join(matched_words),
                    "transcript": whisper_transcript
                })

    # Save results to CSV with UTF-8 encoding
    if contain_results:
        contain_df = pd.DataFrame(contain_results)
        contain_df.to_csv(osp.join(output_dir, "contain_cases.csv"), index=False, encoding="utf-8")
        print(f"Saved contain cases to {osp.join(output_dir, 'contain_cases.csv')}")
    if match_results:
        match_df = pd.DataFrame(match_results)
        match_df.to_csv(osp.join(output_dir, "match_cases.csv"), index=False, encoding="utf-8")
        print(f"Saved match cases to {osp.join(output_dir, 'match_cases.csv')}")
    
    if not match_results and not contain_results:
        print("No common hallucinations found.")
    
    # Remove transcription windows based on the identified hallucinations
    if args.execute_removal:
        # Collect all transcription file paths that need to be removed
        trans_fpaths_to_remove = set()
        for result in contain_results + match_results:
            trans_fpaths_to_remove.add(result["trans_fpath"])

        if not trans_fpaths_to_remove:
            print("No transcription files identified for TSV path removal.")
            return

        print(f"Total paths to remove from TSV: {len(trans_fpaths_to_remove)}")

        # Update the original TSV file by removing paths whose transcriptions were identified with hallucinations
        updated_audio_fpaths = []
        for audio_fpath, trans_fpath in zip(audio_fpaths, trans_fpaths):
            if trans_fpath not in trans_fpaths_to_remove:
                updated_audio_fpaths.append(audio_fpath)
        
        # Write the updated audio file paths back to the original TSV file
        with open(args.original_tsv, "w", encoding="utf-8") as f:
            f.write(root + "\n")
            for audio_fpath in updated_audio_fpaths:
                relative_path = osp.relpath(audio_fpath, root)
                f.write(f"{relative_path}\n")
        print(f"Updated original TSV file by removing paths with hallucinations: {args.original_tsv}")

if __name__ == "__main__":
    main()
