
import argparse
import os
from xmlrpc.client import Boolean
from shared.constants import STATIC_CREATED_FOLDERS, DEFAULT_TEST_RESULTS_DIR
from shared.gcp import download_file_from_bucket, download_folder_from_bucket, BUCKET
from shared.utils import create_run_file_name, extract_number
from scripts.create_graphs import parse_testing_data
from shutil import rmtree

def get_best_run_by_test_accuracy_from_gcp(verbose=False):
    tmp_folder = "tmp"
    download_folder_from_bucket(BUCKET, DEFAULT_TEST_RESULTS_DIR, tmp_folder)
    files = [os.path.join(tmp_folder, f) for f in os.listdir(tmp_folder)]
    best_acc, best_acc_name = 0, ""
    for f in files:
        data = parse_testing_data(f)
        acc = data["Testing"]["binary_accuracy"][-1]
        if verbose:
            print(f"file: {f} -- acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            best_acc_name = f
    if verbose:
        print(f"best acc: {best_acc} ({best_acc_name})")
    run_number = extract_number(best_acc_name)
    rmtree(tmp_folder)
    return run_number
    
    
    


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--run_number", help="run number you wan the results from", default=None, type=int)
    arg_parser.add_argument("-v", "--view", help="see the best accruacy of all test files", required=False, default=False, action="store_true")
    args = arg_parser.parse_args()
    if args.view:
        get_best_run_by_test_accuracy_from_gcp(verbose=True)
        exit()
    if args.run_number is None:
        print("error: the following arguments are required: -r/--run_number")
        exit()
    run_file = create_run_file_name(args.run_numnber)
    for resource, is_file in STATIC_CREATED_FOLDERS.items():
        path = os.path.join(resource, run_file)
        if is_file:
            path += ".txt"
            print(f"downloading: {path}")
            download_file_from_bucket(BUCKET, path)
        else:
            print(f"downloading: {path}")
            download_folder_from_bucket(BUCKET, path)
        print(f"finished download: {resource}")