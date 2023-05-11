import os
import tensorflow as tf
import argparse
import re
from shared.utils import extract_number, find_best_accuracy_from_saved_models
from shared.constants import DEFAULT_SAVED_MODELS_DIR, DEFAULT_TEST_RESULTS_DIR

OUTPUT_DIR = 'app/IsThatSanta/IsThatSanta/models'

def convert_to_tflite(saved_model_dir, best_run, save_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    saved_model_path = os.path.join(save_path, f"run_{best_run}.tflite")
    if os.path.exists(saved_model_path):
        os.remove(saved_model_path)
    with open(saved_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TensorFlow SavedModel format and saved as '{saved_model_path}'")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to TF Lite format")
    parser.add_argument("-m", "--model", type=int, help="model to use.", default=None, required=False)
    return parser.parse_args()

def get_best_saved_model_dir(dir, run_number):
    dir = os.path.join(dir, f"run_{run_number}")
    max_acc = 0
    max_acc_folder = ""
    folders = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    for model_folder in folders:
        file_acc = float(re.findall(
            r"[+-]?\d+\.\d+", model_folder.split("acc")[-1])[0])
        if file_acc > max_acc:
            max_acc = file_acc
            max_acc_folder = model_folder
    return max_acc_folder

def determine_best_test_run():
    def get_accuracy(file_line):
        pattern = r"binary_accuracy:\s([0-9.]+)"
        match = re.search(pattern, file_line)
        if match:
            return float(match.group(1))
        return None
    if not os.path.isdir(DEFAULT_TEST_RESULTS_DIR):
        return None
    files = os.listdir(DEFAULT_TEST_RESULTS_DIR)
    file_paths = [os.path.join(DEFAULT_TEST_RESULTS_DIR, f) for f in files]
    max_acc, max_file_number = -1, None
    for file in file_paths:
        with open(file, 'r') as f:
            last_line = None
            for line in f:
                last_line = line
            accuracy = get_accuracy(last_line)
            if accuracy > max_acc:
                max_acc = accuracy
                max_file_number = extract_number(file)
    return max_file_number

def main():
    args = parse_args()
    best_run = args.model
    if best_run:
        path = os.path.join(DEFAULT_SAVED_MODELS_DIR, f"run_{best_run}")
        best_model_folder, _ = find_best_accuracy_from_saved_models(path)
    else:
        best_run = determine_best_test_run()
        path = os.path.join(DEFAULT_SAVED_MODELS_DIR, f"run_{best_run}")
        best_model_folder, _ = find_best_accuracy_from_saved_models(path)
    if best_model_folder is None:
        print("an error occured in parsing")
        return
    convert_to_tflite(best_model_folder, best_run, OUTPUT_DIR)
    