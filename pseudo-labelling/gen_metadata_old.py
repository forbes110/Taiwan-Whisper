# #!/usr/bin/env python3
# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# """
# Data pre-processing: build vocabularies and binarize training data.
# """

# import argparse
# import glob
# import os
# import random
# import tqdm
# import soundfile as sf
# import multiprocessing as mp

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "root", metavar="DIR", help="root directory containing flac) files to index"
#     )
#     parser.add_argument(
#         "--valid-percent",
#         default=0.0,
#         type=float,
#         metavar="D",
#         help="percentage of data to use as validation set (between 0 and 1)",
#     )
#     parser.add_argument(
#         "--output_fname", default="train", type=str, metavar="NAME", help="output fname"
#     )
#     parser.add_argument(
#         "--dest", default=".", type=str, metavar="DIR", help="output directory"
#     )
#     parser.add_argument(
#         "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
#     )
#     parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
#     parser.add_argument(
#         "--path-must-contain",
#         default=None,
#         type=str,
#         metavar="FRAG",
#         help="if set, path must contain this substring for a file to be included in the manifest",
#     )
#     parser.add_argument(
#         "--path-must-not-contain",
#         default=None,
#         type=str,
#         # metavar="FRAG",
#         help="if set, path must contain this substring for a file to be included in the manifest",
#     )
#     parser.add_argument(
#         "--sort",
#         default=False,
#         action="store_true",
#         help="sort the list of files before writing them to the manifest",
#     )
#     parser.add_argument(
#         "--get-frames",
#         default=False,
#         action="store_true",
#         help="get the number of frames in the audio file",
#     )
#     return parser

# def get_frames(file_path):
    
#     frames = sf.info(file_path).frames
#     return file_path, frames

# def main(args):
#     assert args.valid_percent >= 0 and args.valid_percent <= 1.0

#     if not os.path.exists(args.dest):
#         os.makedirs(args.dest)

#     dir_path = os.path.realpath(args.root)
#     search_path = os.path.join(dir_path, "**/*." + args.ext)
#     rand = random.Random(args.seed)

#     output_fname = args.output_fname
#     valid_fname = "valid" if output_fname == "train" else f"{output_fname}-valid"
#     valid_f = (
#         open(os.path.join(args.dest, f"{valid_fname}.tsv"), "w")
#         if args.valid_percent > 0
#         else None
#     )

#     with open(os.path.join(args.dest, f"{output_fname}.tsv"), "w") as train_f:
#         print(dir_path, file=train_f)

#         if valid_f is not None:
#             print(dir_path, file=valid_f)
        
        
#         file_paths = [os.path.realpath(fname) for fname in glob.iglob(search_path, recursive=True)]
#         if args.get_frames:
#             pool = mp.Pool(processes=64)
#             file_paths_and_frames = list(tqdm.tqdm(pool.imap_unordered(get_frames, file_paths), total=len(file_paths)))
#         else:
#             file_paths_and_frames = [(file_path, None) for file_path in file_paths]
#         if args.sort:
#             file_paths_and_frames = sorted(file_paths_and_frames, key=lambda x: x[0])

#         for file_path, frame in file_paths_and_frames:
#             if args.path_must_contain and args.path_must_contain not in file_path:
#                 continue
#             if args.path_must_not_contain and args.path_must_not_contain in file_path:
#                 continue

#             dest = train_f if rand.random() > args.valid_percent else valid_f
#             if frame is None:
#                 print(os.path.relpath(file_path, dir_path), file=dest)
#             else:
#                 print(
#                     "{}\t{}".format(os.path.relpath(file_path, dir_path), frame), file=dest
#                 )
#     if valid_f is not None:
#         valid_f.close()

#     print(f"generate metadata(of paths) at {args.dest}/{output_fname}.tsv")

# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args)


import argparse
import glob
import os
import random
import tqdm
import soundfile as sf
import multiprocessing as mp
from typing import Tuple, List, Optional

def get_parser():
    parser = argparse.ArgumentParser(description="音訊檔案處理與資料集分割工具")
    parser.add_argument(
        "root", metavar="DIR", help="包含音訊檔案的根目錄"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="用作驗證集的資料比例 (0-1 之間)"
    )
    parser.add_argument(
        "--output_fname", 
        default="train", 
        type=str, 
        metavar="NAME", 
        help="輸出檔案名稱"
    )
    parser.add_argument(
        "--dest", 
        default=".", 
        type=str, 
        metavar="DIR", 
        help="輸出目錄"
    )
    parser.add_argument(
        "--ext", 
        default="flac", 
        type=str, 
        metavar="EXT", 
        help="要處理的檔案副檔名"
    )
    parser.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        metavar="N", 
        help="隨機種子"
    )
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="檔案路徑必須包含的字串"
    )
    parser.add_argument(
        "--path-must-not-contain",
        default=None,
        type=str,
        help="檔案路徑不得包含的字串"
    )
    parser.add_argument(
        "--sort",
        default=False,
        action="store_true",
        help="是否對檔案列表進行排序"
    )
    parser.add_argument(
        "--get-frames",
        default=False,
        action="store_true",
        help="是否獲取音訊檔案的幀數"
    )
    parser.add_argument(
        "--num-workers",
        default=None,
        type=int,
        help="多處理的工作程序數量，預設使用 CPU 核心數"
    )
    return parser

def process_file(args: Tuple[str, argparse.Namespace, str, random.Random]) -> Optional[Tuple[str, str, bool]]:
    """
    處理單個音訊檔案
    
    Parameters:
    - args: (檔案路徑, 參數物件, 根目錄路徑, 隨機數產生器)
    
    Returns:
    - (相對路徑, 輸出字串, 是否為訓練集) 或 None（處理失敗時）
    """
    file_path, params, dir_path, rand = args
    
    try:
        # 檢查路徑條件
        if params.path_must_contain and params.path_must_contain not in file_path:
            return None
        if params.path_must_not_contain and params.path_must_not_contain in file_path:
            return None
            
        # 取得相對路徑
        rel_path = os.path.relpath(file_path, dir_path)
        
        # 如果需要獲取幀數
        if params.get_frames:
            try:
                frames = sf.info(file_path).frames
                output_str = f"{rel_path}\t{frames}"
            except Exception as e:
                print(f"無法讀取檔案 {file_path} 的幀數: {str(e)}")
                return None
        else:
            output_str = rel_path
            
        # 決定是否為訓練集
        is_train = rand.random() > params.valid_percent
        
        return (rel_path, output_str, is_train)
        
    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {str(e)}")
        return None

def main(args):
    # 參數驗證
    assert 0 <= args.valid_percent <= 1.0, "valid-percent 必須在 0 和 1 之間"
    
    # 確保輸出目錄存在
    os.makedirs(args.dest, exist_ok=True)
    
    # 設定路徑和隨機數產生器
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)
    
    # 設定輸出檔案名稱
    output_fname = args.output_fname
    valid_fname = "valid" if output_fname == "train" else f"{output_fname}-valid"
    
    # 收集所有檔案路徑
    file_paths = [
        os.path.realpath(fname)
        for fname in glob.iglob(search_path, recursive=True)
    ]
    
    if not file_paths:
        print(f"警告：在 {args.root} 中未找到 .{args.ext} 檔案")
        return
        
    # 設定工作程序數量
    num_workers = args.num_workers or int(mp.cpu_count()*0.9)
    num_workers = min(num_workers, len(file_paths))
    
    # 準備處理參數
    process_args = [
        (path, args, dir_path, random.Random(args.seed))
        for path in file_paths
    ]
    
    # 使用進程池進行並行處理
    train_files = []
    valid_files = []
    
    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm.tqdm(
            pool.imap_unordered(process_file, process_args),
            total=len(file_paths),
            desc="處理檔案中"
        ):
            if result is None:
                continue
            rel_path, output_str, is_train = result
            if is_train:
                train_files.append(output_str)
            else:
                valid_files.append(output_str)
    
    # 如果需要排序
    if args.sort:
        train_files.sort()
        valid_files.sort()
    
    # 寫入訓練集檔案
    train_path = os.path.join(args.dest, f"{output_fname}.tsv")
    with open(train_path, "w") as train_f:
        print(dir_path, file=train_f)
        for file_info in train_files:
            print(file_info, file=train_f)
    
    # 如果需要，寫入驗證集檔案
    if args.valid_percent > 0:
        valid_path = os.path.join(args.dest, f"{valid_fname}.tsv")
        with open(valid_path, "w") as valid_f:
            print(dir_path, file=valid_f)
            for file_info in valid_files:
                print(file_info, file=valid_f)
    
    print(f"已完成處理：")
    print(f"- 訓練集：{len(train_files)} 個檔案")
    if args.valid_percent > 0:
        print(f"- 驗證集：{len(valid_files)} 個檔案")
    print(f"輸出檔案位置：{args.dest}/{output_fname}.tsv")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
