import argparse
import os
import subprocess
from zipfile import ZipFile
from dotenv import load_dotenv
from shutil import rmtree
import json
from shared.gcp import upload_directory_to_bucket, BUCKET, download_folder_from_bucket
from shared.constants import DEFAULT_DATA_DIR

load_dotenv()

USER_NAME = os.environ.get('KAGGLE_USERNAME')
KEY = os.environ.get('KAGGLE_KEY')

OLD_DATASET = "https://www.kaggle.com/datasets/ianmaloney/santa-images"
NEW_DATASET = "https://www.kaggle.com/datasets/ianmaloney/updated-santa"

def create_kaggle_json_file(username, password):
    data = {'username': username, 'key': password}
    dir_path = os.path.join(f"{os.sep}root", ".kaggle")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    output_file = os.path.join(dir_path, "kaggle.json")
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as f:
        json.dump(data, f)

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--directory", help="Directory to save data to", default=DEFAULT_DATA_DIR)
    arg_parser.add_argument("-u", "--username", help="user name for kaggle api", default=USER_NAME, required=False)
    arg_parser.add_argument("-p", "--password", help="password for kaggle api", default=KEY, required=False)
    
    return arg_parser.parse_args()

def wipe_dir(directory):
    if os.path.exists(directory):
        rmtree(directory)

def unzip_data(zip_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    
def download_kaggle_dataset(dataset_url, directory):
    dataset_identifier = dataset_url.split("/")[-1]
    owner_slug = dataset_url.split("/")[-2]
    try:
        subprocess.run(["kaggle", "datasets", "download", f"{owner_slug}/{dataset_identifier}"])
        return f"{dataset_identifier}.zip"
    except Exception as e:
        print(e)
        return None

def upload_data_to_gcp():
    upload_directory_to_bucket(BUCKET, DEFAULT_DATA_DIR)

def download_data_from_gcp():
    download_folder_from_bucket(BUCKET, DEFAULT_DATA_DIR)

def main():
    args = parse_args()
    directory = args.directory
    username, password = args.username, args.password
    wipe_dir(directory)
    print(f"USERNAME: {username}\nPASSWORD: {password}")
    create_kaggle_json_file(username, password)
    dataset_url = NEW_DATASET
    zip_path = download_kaggle_dataset(dataset_url, directory)
    if zip_path is None:
        print("could not get the zip path")
        return
    unzip_data(zip_path)
    os.remove(zip_path)
