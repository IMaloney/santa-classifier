from google.cloud import storage
import os

BUCKET = 'is_that_santa'
PATH_TO_JSON = 'gcp-santa-key.json'
    
def create_folder_in_gcp_bucket(bucket_name, folder_path):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(folder_path)
    if not blob.exists():
        blob.upload_from_string("")
        return True
    return False

def upload_file_to_bucket(bucket_name, source_file):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file)
    blob.upload_from_filename(source_file)
    return True

def upload_directory_to_bucket(bucket_name, source_directory):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            blob = bucket.blob(file_path)
            blob.upload_from_filename(file_path)

def upload_to_gcp(paths):
    for path in paths:
        if os.path.isdir(path):
            upload_directory_to_bucket(BUCKET, path)
        else:
            upload_file_to_bucket(BUCKET, path)

def download_file_from_bucket(bucket_name, source_file):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file)
    blob.download_to_filename(source_file)

def download_folder_from_bucket(bucket_name, folder_path, local_path=""):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    for blob in blobs:
        local_file_path = os.path.join(local_path, blob.name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)

def create_bucket(bucket_name, project_id, storage_class, location):
    storage_client = storage.Client.from_service_account_json(PATH_TO_JSON)
    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = storage_class
    bucket.create(location=location)