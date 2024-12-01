# make psuedo label from simplified chinese to traditional chinese
import os
import sys
import argparse
from opencc import OpenCC
import chardet
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

TEXT_FILE_EXTENSIONS = {'.csv', '.tsv'}

def detect_encoding(file_path):
   with open(file_path, 'rb') as f:
       raw = f.read(100000)
   result = chardet.detect(raw)
   return result['encoding']

def convert_file(args):
   file_path, overwrite = args
   try:
       converter = OpenCC('s2t')
       encoding = detect_encoding(file_path)
       if encoding is None:
           return f"Skipping {file_path}: Unable to detect encoding."

       with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
           content = f.read()
       
       converted = converter.convert(content)

       if overwrite:
           with open(file_path, 'w', encoding=encoding, errors='ignore') as f:
               f.write(converted)
           return f"Converted and overwritten: {file_path}"
       else:
           new_file = f"{file_path}.trad"
           with open(new_file, 'w', encoding=encoding, errors='ignore') as f:
               f.write(converted)
           return f"Converted and saved as: {new_file}"

   except Exception as e:
       return f"Error processing {file_path}: {e}"

def get_all_files(directory):
   files = []
   for root, _, filenames in os.walk(directory):
       for filename in filenames:
           if os.path.splitext(filename)[1].lower() in TEXT_FILE_EXTENSIONS:
               files.append(os.path.join(root, filename))
   return files

def main():
   parser = argparse.ArgumentParser(description="Convert Simplified Chinese to Traditional Chinese")
   parser.add_argument('--path', type=str, help="Path to file or directory")
   parser.add_argument('--output', choices=['overwrite', 'new'], default='overwrite')
   parser.add_argument('--workers', type=int, default=int(os.cpu_count()*3/4), help="Number of workers")
   args = parser.parse_args()

   if not os.path.exists(args.path):
       print(f"Error: Path '{args.path}' does not exist.")
       sys.exit(1)

   files_to_process = []
   if os.path.isfile(args.path):
       if os.path.splitext(args.path)[1].lower() in TEXT_FILE_EXTENSIONS:
           files_to_process.append(args.path)
   else:
       files_to_process = get_all_files(args.path)

   if not files_to_process:
       print("No files to process")
       return

   with ProcessPoolExecutor(max_workers=args.workers) as executor:
       futures = [executor.submit(convert_file, (f, args.output == 'overwrite')) 
                 for f in files_to_process]
       
       for future in tqdm(futures, total=len(files_to_process), desc="Converting files"):
           print(future.result())

if __name__ == "__main__":
   main()


