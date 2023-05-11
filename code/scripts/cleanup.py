import os
from shutil import rmtree
import re
import argparse
from shared.constants import STATIC_CREATED_FOLDERS

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--all", help="deletes everything", default=False, required=False, action="store_true")
    arg_parser.add_argument("-r", "--run", help="number of the run you want to delete", type=int, required=False, default=0)
    return arg_parser.parse_args()    

def delete_run(run_number):
    folders = [f for f in STATIC_CREATED_FOLDERS if os.path.isdir(f)]
    for folder in folders:
        print(f"found folder: {folder}")
        files = os.listdir(folder)
        for f in files:
            try:
                file_number = int(re.search(r'_(\d+)', f).group(1))
                if file_number != run_number:
                    continue
                path_to_file = os.path.join(folder, f)
                if os.path.isdir(path_to_file):
                    print(f"deleting directory: {path_to_file}")
                    rmtree(path_to_file)
                else:
                    print(f"deleting file: {path_to_file}")
                    os.remove(path_to_file)                 
            except (ValueError, AttributeError):
                pass
    print("done")     

def remove_folders():
    for folder in STATIC_CREATED_FOLDERS:
        if os.path.isdir(folder):
            print(f"deleting folder: {folder}")
            rmtree(folder)
        else:
            print(f"folder: {folder} does not exist, skipping...")
        