import os
import tensorflow as tf
import model.hyperparameters as hp
import numpy as np
from sklearn.utils import class_weight

class SantaDataset():
    def __init__(self, data_directory, ):
        self.data_directory = data_directory
        self.labels = ["" for _ in range(hp.num_classes)]
        self.classes = [0, 1]
        self.train_data, self.validation_data = self.get_train_data()
        self.test_data = self.get_test_data()
        self.class_weights = None
        self.create_class_weights()
        
    def create_class_weights(self):
        weights = class_weight.compute_class_weight("balanced", classes=np.unique(self.train_data.classes), y=self.train_data.classes)
        self.class_weights = dict(enumerate(weights))
        
    def setup_labels(self, path, data_gen):
        unordered_classes = []
        for dir_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, dir_name)):
                unordered_classes.append(dir_name)
        for img_class in unordered_classes:
            self.labels[data_gen.class_indices[img_class]] = img_class
           
    def preprocess_data(self, img):        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    
    def get_train_data(self):
        path = os.path.join(self.data_directory, "train/")
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            brightness_range=[0.4, 0.6],
            channel_shift_range=0.85,
            horizontal_flip=True, 
            rotation_range=45, 
            vertical_flip=True, 
            shear_range=.8, 
            zoom_range=.5,
            height_shift_range=35,
            width_shift_range=35,
            rescale=1.0/255.0,
            samplewise_center=True,
            validation_split=hp.validation_split,
            preprocessing_function=self.preprocess_data
        )
        
        training_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.image_size, hp.image_size),
            shuffle=True, 
            class_mode='binary', 
            subset="training",
            batch_size=hp.batch_size
        )
        validation_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.image_size, hp.image_size),
            shuffle=True, 
            class_mode='binary', 
            subset="validation",
            batch_size=hp.batch_size
        )
        self.setup_labels(path, training_gen)
        return training_gen, validation_gen
        
    def get_test_data(self):
        path = os.path.join(self.data_directory, "test/")
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self.preprocess_data)
        data_gen = data_gen.flow_from_directory(
            path, 
            target_size=(hp.image_size, hp.image_size), 
            class_mode='binary', 
            batch_size=hp.test_batch_size, 
            shuffle=False, 
        )
        return data_gen
        
        