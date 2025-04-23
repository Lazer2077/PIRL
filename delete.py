import os
import shutil

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def remove_small_dirs(parent_dir, threshold_bytes=400):
    for entry in os.listdir(parent_dir):
        entry_path = os.path.join(parent_dir, entry)
        if os.path.isdir(entry_path):
            size = get_dir_size(entry_path)
            if size < threshold_bytes:
                print(f"Deleting folder: {entry_path} (size: {size} bytes)")
                shutil.rmtree(entry_path)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='/mnt/d/RL/LogTmp')
args = parser.parse_args()

target_path = args.target_path
remove_small_dirs(target_path)
