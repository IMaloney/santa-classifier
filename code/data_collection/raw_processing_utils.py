import os
from data_collection.data_collection_utils import calculate_phash, get_images

def find_and_remove_similar_images(folder_path, threshold, verbose = False):
    image_hashes = {}
    removed_images = 0
    files = get_images(folder_path)
    for filename in files:
        image_hash = calculate_phash(filename)
        for stored_hash, stored_path in image_hashes.items():
            hash_difference = image_hash - stored_hash
            if hash_difference <= threshold:
                if verbose:
                    print(f"Visually similar images: {filename} (removed) and {stored_path} (hash difference: {hash_difference})")
                os.remove(filename)
                removed_images += 1
                break
        else:
            image_hashes[image_hash] = filename
    if verbose:
        print(f"# removed images: {removed_images}")

def rename_images(folder_path, new_name, verbose = False):  
    image_files = get_images(folder_path)
    file_count = 1
    for old_file_path in image_files:
        new_file_name = f"{new_name}_{file_count}.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        if verbose:
            print(f"Renamed {old_file_path} to {new_file_path}")
        file_count += 1
    print("Done!")