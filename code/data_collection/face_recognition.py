import cv2
import os
from natsort import natsorted
from shared.utils import get_images

CONFIG_FILE = "caffe/deploy.prototxt"
MODEL_FILE = "caffe/mobilenet_iter_73000.caffemodel"

def find_person(image_path, threshold=0.2):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (300, 300))
    model = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)
    model.setInput(blob)
    detections = model.forward()
    people_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if class_id == 15 and confidence > threshold:
            people_count += 1
    return people_count

def find_images_with_more_than_one_face(folder_path):
    ct = 0
    files_list = list()
    files = get_images(folder_path)
    files = natsorted(files)
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        num_faces = find_person(file_path)
        if num_faces > 1:
            print(f"file: {file_path} has more than one face")
            ct += 1
            files_list.append(file_path)
    print(f"Done: discovered {ct} images with more than one face")
    return files_list

def find_images_with_no_faces(folder_path):
    ct = 0
    files_list = list()
    files = get_images(folder_path)
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        num_faces = find_person(file_path)
        if num_faces == 0:
            print(f"file: {file_path} has no faces")
            ct += 1
            files_list.append(file_path)
    print(f"Done: discovered {ct} images with no face")
    return files_list