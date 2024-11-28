# from transformers.models.whisper.english_normalizer import BasicTextNormalizer
# import argparse
# import os.path as osp
# import os
# import re
# import pandas as pd

# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(description="Remove common hallucinations from transcriptions.")
#     parser.add_argument(
#         "--original_tsv",
#         type=str,
#         default="/mnt/audio_paths.tsv",
#         help="Path to TSV file containing audio file paths."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/mnt/common_hallucination_caught/FTV_selected",
#         help="Directory to save the results."
#     )
#     parser.add_argument(
#         "--execute_removal",
#         action='store_true',
#         help="Flag to remove paths from the TSV file if they contain hallucinations, without deleting audio files."
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
#         "會為大家說明",
#     ]

#     hallucination_contain_list = [
#         "請不吝",
#         # r"(?<!\w)org(?!\w)",  # Ensure examples like "organization" are not removed
#         "點贊",
#         "點讚",
#         "字幕提供",
#         "支持明鏡",
#         "點點欄目",
#         "會為大家說明",
#         "Thank you very much.",
#         "Thank you for watching my video."
#     ]
#     # Initialize lists to store results
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
#         print("No common hallucinations found.")
    
#     # Remove transcription windows based on the identified hallucinations
#     if args.execute_removal:
#         # Collect all transcription file paths that need to be removed
#         trans_fpaths_to_remove = set()
#         for result in contain_results + match_results:
#             trans_fpaths_to_remove.add(result["trans_fpath"])

#         if not trans_fpaths_to_remove:
#             print("No transcription files identified for TSV path removal.")
#             return

#         print(f"Total paths to remove from TSV: {len(trans_fpaths_to_remove)}")

#         # Update the original TSV file by removing paths whose transcriptions were identified with hallucinations
#         updated_audio_fpaths = []
#         for audio_fpath, trans_fpath in zip(audio_fpaths, trans_fpaths):
#             if trans_fpath not in trans_fpaths_to_remove:
#                 updated_audio_fpaths.append(audio_fpath)
        
#         # Write the updated audio file paths back to the original TSV file
#         with open(args.original_tsv, "w", encoding="utf-8") as f:
#             f.write(root + "\n")
#             for audio_fpath in updated_audio_fpaths:
#                 relative_path = osp.relpath(audio_fpath, root)
#                 f.write(f"{relative_path}\n")
#         print(f"Updated original TSV file by removing paths with hallucinations: {args.original_tsv}")

# if __name__ == "__main__":
#     main()


from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse
import os.path as osp
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
import logging
from typing import List, Dict, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        help="Flag to remove paths from the TSV file if they contain hallucinations."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use for processing."
    )
    return parser.parse_args()

class HallucationDetector:
    def __init__(self):
        # Initialize lists for hallucination patterns
        self.hallucination_match_list = [
            "Okay.",
            "...",
            ".",
            "Mm.",
            "會為大家說明",
        ]

        self.hallucination_contain_list = [
            "請不吝",
            "點贊",
            "點讚",
            "字幕提供",
            "支持明鏡",
            "點點欄目",
            "會為大家說明",
            "Thank you very much.",
            "Thank you for watching my video."
        ]
        
        # Initialize thread-safe collections for results
        self.contain_results: List[Dict] = []
        self.match_results: List[Dict] = []
        self.results_lock = Lock()
        
        # Initialize the normalizer (shared across threads)
        self.normalizer = BasicTextNormalizer()

    def process_transcription(self, trans_fpath: str) -> None:
        """
        Process a single transcription file to detect hallucinations.
        This method is designed to be thread-safe.
        """
        if not osp.exists(trans_fpath):
            logging.warning(f"Transcription file not found: {trans_fpath}")
            return

        try:
            with open(trans_fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                logging.warning(f"Empty transcription file: {trans_fpath}")
                return

            # Process the transcript
            whisper_transcript = self._prepare_transcript(lines)
            
            # Check for hallucinations
            contain_matches = self._check_contains(whisper_transcript)
            match_matches = self._check_matches(whisper_transcript)

            # Thread-safe addition of results
            with self.results_lock:
                if contain_matches:
                    self.contain_results.append({
                        "trans_fpath": trans_fpath,
                        "matched_keywords": ', '.join(contain_matches),
                        "transcript": whisper_transcript
                    })
                
                if match_matches:
                    self.match_results.append({
                        "trans_fpath": trans_fpath,
                        "matched_words": ', '.join(match_matches),
                        "transcript": whisper_transcript
                    })

        except Exception as e:
            logging.error(f"Error processing {trans_fpath}: {str(e)}")

    def _prepare_transcript(self, lines: List[str]) -> str:
        """Prepare and normalize the transcript text."""
        # Remove special tokens and clean up the transcript
        whisper_transcript = lines[0].strip().split("<|endoftext|>")[0]
        whisper_transcript = whisper_transcript.split("<|continued|>")[0]
        
        # Remove timestamp tokens
        timestamp_tokens = re.findall(r"<\|\d{1,2}\.\d{2}\|>", whisper_transcript)
        for st in timestamp_tokens:
            whisper_transcript = whisper_transcript.replace(st, ' ')
            
        whisper_transcript = whisper_transcript.strip().replace('  ', ' ')
        return self.normalizer(whisper_transcript)

    def _check_contains(self, transcript: str) -> List[str]:
        """Check for contained hallucinations in the transcript."""
        return [keyword for keyword in self.hallucination_contain_list 
                if re.search(keyword, transcript)]

    def _check_matches(self, transcript: str) -> List[str]:
        """Check for exact matches in the transcript."""
        words = re.findall(r'\b\w+\b|\.\.\.|[^\s\w]', transcript)
        return [word for word in words if word in self.hallucination_match_list]

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")
    
    # Read the TSV file
    with open(args.original_tsv, "r", encoding="utf-8") as f:
        root = f.readline().strip()
        audio_subfpaths = [l.strip() for l in f.readlines()]
        audio_fpaths = [osp.join(root, audio_subfpath) for audio_subfpath in audio_subfpaths]
        trans_fpaths = [audio_fpath.replace('flac', 'txt') for audio_fpath in audio_fpaths]

    # Initialize the detector
    detector = HallucationDetector()
    
    # Process files using thread pool
    logging.info(f"Starting processing with {args.num_threads} threads")
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all transcription files for processing
        executor.map(detector.process_transcription, trans_fpaths)
    
    # Save results
    if detector.contain_results:
        contain_df = pd.DataFrame(detector.contain_results)
        contain_output = osp.join(args.output_dir, "contain_cases.csv")
        contain_df.to_csv(contain_output, index=False, encoding="utf-8")
        logging.info(f"Saved contain cases to {contain_output}")
    
    if detector.match_results:
        match_df = pd.DataFrame(detector.match_results)
        match_output = osp.join(args.output_dir, "match_cases.csv")
        match_df.to_csv(match_output, index=False, encoding="utf-8")
        logging.info(f"Saved match cases to {match_output}")
    
    if not detector.match_results and not detector.contain_results:
        logging.info("No common hallucinations found.")
    
    # Handle removal if requested
    if args.execute_removal:
        _handle_removal(args, detector, root, audio_fpaths, trans_fpaths)

def _handle_removal(args, detector, root, audio_fpaths, trans_fpaths):
    """Handle the removal of identified hallucination cases from the TSV file."""
    # Collect all transcription file paths that need to be removed
    trans_fpaths_to_remove = {result["trans_fpath"] 
                             for result in detector.contain_results + detector.match_results}

    if not trans_fpaths_to_remove:
        logging.info("No transcription files identified for TSV path removal.")
        return

    logging.info(f"Total paths to remove from TSV: {len(trans_fpaths_to_remove)}")

    # Update and write back to the TSV file
    updated_audio_fpaths = [audio_fpath for audio_fpath, trans_fpath in zip(audio_fpaths, trans_fpaths)
                           if trans_fpath not in trans_fpaths_to_remove]
    
    with open(args.original_tsv, "w", encoding="utf-8") as f:
        f.write(root + "\n")
        for audio_fpath in updated_audio_fpaths:
            relative_path = osp.relpath(audio_fpath, root)
            f.write(f"{relative_path}\n")
    
    logging.info(f"Updated original TSV file: {args.original_tsv}")

if __name__ == "__main__":
    main()