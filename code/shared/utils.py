import os
import re
from shared.constants import STATIC_CREATED_FOLDERS
from natsort import natsorted
import string 
import random

def create_run_file_name(number):
    return f'run_{number}'

def extract_number(file_name):
    pattern = r"run_(\d+)"
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(1))
    return None
    
def determine_run_num(directory_path):
    files = list()
    for dir_name in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, dir_name)):
            files.append(dir_name)
    run_num = 1
    files = natsorted(files)
    pattern = r'_(\d+)'
    for f in files:
        try:
            file_number = int(re.search(pattern, f).group(1))
            if file_number != run_num:
                return run_num
            run_num += 1                
        except (ValueError, AttributeError):
            pass
    return run_num

def get_run_num(directory):
    if os.path.isdir(directory):
        return determine_run_num(directory)
    print(f"{directory} dir doesn't exist, creating it now.")
    os.makedirs(directory)
    return 1 

def random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def get_images(directory):
    def is_image(filename):
        return filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))    
    files = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith(".") and is_image(f)]
    return natsorted(files)

def find_best_accuracy_from_saved_models(directory_path):
    max_acc = 0
    max_acc_folder = ''
    folders = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
    pattern = r"[+-]?\d+\.\d+"
    for model_folder in folders:
        file_acc = float(re.findall(
            pattern, model_folder.split("acc")[-1])[0])
        if file_acc > max_acc:
            max_acc = file_acc
            max_acc_folder = model_folder
    return max_acc_folder, max_acc

def update_file(path, line):
        with open(path, 'a') as file:
            file.write(line) 
    
def create_all_folders():
    for folder in STATIC_CREATED_FOLDERS.keys():
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
def make_nested_folder(dir, run_num):
    dir_name = create_run_file_name(run_num)
    path = os.path.join(dir, dir_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def calculate_precision_recall_f1(tp, fp, fn):
    prec = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0 if (prec + recall) == 0 else 2 * (prec * recall) / (prec + recall)
    return prec, recall, f1
    