from PIL import Image
import imagehash
import cv2
import os
import random
from natsort import natsorted
from shared.utils import get_images


def calculate_phash(image_path):
    with Image.open(image_path) as img:
        return imagehash.phash(img)

def shuffle_images(directory, verbose=False):
    files = get_images(directory)
    sorted_files = natsorted(files)
    random.shuffle(files)
    temp_files = [os.path.join(directory, f'temp_{i}.tmp') for i in range(len(files))]
    
    for original, temp in zip(files, temp_files):
        os.rename(original, temp)
        
    for shuffled, original_name in zip(temp_files, sorted_files):
        os.rename(shuffled, original_name)
    if verbose:
        print("Done shuffling images")

def prompt_delete(files_list):
    ct = 0
    for file_path in files_list:
        img = cv2.imread(file_path)
        cv2.imshow('Image', img)
        cv2.waitKey(1)
        
        choice = input("Would you like to delete this image? (y/n): ")
        cv2.destroyAllWindows()
        if choice.lower() == "y":
            os.remove(file_path)
            print(f"{ct}: {file_path} deleted.")
        else:
            print(f"{ct}: {file_path} not deleted.")
        ct += 1