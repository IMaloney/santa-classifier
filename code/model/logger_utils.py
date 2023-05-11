"""
A small portion borrowed from HW5
CS1430 - Computer Vision
Brown University
"""

import io
import os
import re
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import model.hyperparameters as hp
from collections import deque
from shared.utils import find_best_accuracy_from_saved_models, update_file, calculate_precision_recall_f1

def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

class ImageLabelingLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a plot of test images and their
    predicted labels for viewing in Tensorboard. """

    def __init__(self, logs_save_dir, dataset):
        super(ImageLabelingLogger, self).__init__()
        self.dataset = dataset
        self.logs_path = logs_save_dir

        print("Done setting up image labeling logger.")
    
    def on_epoch_end(self, epoch, logs=None):
        self.log_image_labels(epoch, logs)

    def log_image_labels(self, epoch_num, logs):
        """ Writes a plot of test images and their predicted labels
        to disk. """

        fig = plt.figure(figsize=(9, 9))
        count_all = 0
        count_misclassified = 0
        
        for batch in self.dataset.train_data:
            misclassified = []
            correct_labels = []
            wrong_labels = []

            for i, image in enumerate(batch[0]):
                plt.subplot(5, 5, min(count_all+1, 25))

                correct_class_idx = batch[1][i]
                probabilities = self.model(np.array([image])).numpy()[0]
                predict_class_idx = np.argmax(probabilities)

                image = np.clip(image, 0., 1.)
                plt.imshow(image, cmap='gray')

                is_correct = correct_class_idx == predict_class_idx

                title_color = 'b' if is_correct else 'r'

                plt.title(
                    self.dataset.labels[predict_class_idx],
                    color=title_color)
                plt.axis('off')                
                # output individual images with wrong labels
                if not is_correct:
                    count_misclassified += 1
                    misclassified.append(image)
                    correct_labels.append(correct_class_idx)
                    wrong_labels.append(predict_class_idx)

                count_all += 1
                
                # ensure there are >= 2 misclassified images
                if count_all >= 25 and count_misclassified >= 2:
                    break

            if count_all >= 25 and count_misclassified >= 2:
                break

        figure_img = plot_to_image(fig)

        file_writer_il = tf.summary.create_file_writer(
            os.path.join(self.logs_path, "image_labels"))

        misclassified_path = os.path.join(self.logs_path, "mislabeled")
        if not os.path.exists(misclassified_path):
            os.makedirs(misclassified_path)
        for correct, wrong, img in zip(correct_labels, wrong_labels, misclassified):
            wrong = self.dataset.labels[wrong.astype(np.int32)]
            correct= self.dataset.labels[correct.astype(np.int32)]
            image_name = wrong + "_predicted" + ".png"
            new_path = os.path.join(misclassified_path, correct)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            plt.imsave(os.path.join(new_path, image_name), img)

        with file_writer_il.as_default():
            tf.summary.image("0 Example Set of Image Label Predictions (blue is correct; red is incorrect)",
                             figure_img, step=epoch_num)
            for label, wrong, img in zip(correct_labels, wrong_labels, misclassified):
                img = tf.expand_dims(img, axis=0)
                tf.summary.image("1 Example @ epoch " + str(epoch_num) + ": " + self.dataset.labels[label.astype(np.int32)] + " misclassified as " + self.dataset.labels[wrong.astype(np.int32)], 
                                 img, step=epoch_num)

def create_template_string(s, loss, acc, tp, tn, fp, fn, prec, recall, f1):
        return f"\t{s}:\n\t\tloss: {loss}\n\t\taccuracy: {acc}\n\t\tTrue Positives: {tp}\n\t\tTrue Negatives: {tn}\n\t\tFalse Positives: {fp}\n\t\tFalse Negatives: {fn}\n\t\tPrecision: {prec}\n\t\tRecall: {recall}\n\t\tF1 Score: {f1}\n"

class ModelLogger(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """
    def __init__(self, info_save_file, model_save_dir):
        super(ModelLogger, self).__init__()
        self.info_save_file = info_save_file
        self.saved_models_dir = model_save_dir

    def create_info_file_line(self, epoch, logs):
        t_acc = logs["binary_accuracy"]
        v_acc = logs["val_binary_accuracy"]
        t_tp = logs["true_positives"]
        v_tp = logs["val_true_positives"]
        t_tn = logs["true_negatives"]
        v_tn = logs["val_true_negatives"]
        t_fp = logs["false_positives"]
        v_fp = logs["val_false_positives"]
        t_fn = logs["false_negatives"]
        v_fn = logs["val_false_negatives"]
        t_loss = logs["loss"]
        v_loss = logs["val_loss"]
        t_prec, t_recall, t_f1 = calculate_precision_recall_f1(t_tp, t_fp, t_fn)
        v_prec, v_recall, v_f1 = calculate_precision_recall_f1(v_tp, v_fp, v_fn)
        epoch_part = f"Epoch: {epoch}\n"
        training_part = create_template_string("Training", t_loss, t_acc, t_tp, t_tn, t_fp, t_fn, t_prec, t_recall, t_f1)
        validation_part = create_template_string("Validation", v_loss, v_acc, v_tp, v_tn, v_fp, v_fn, v_prec, v_recall, v_f1)
        return epoch_part + training_part + validation_part
    
    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, models are saved to saved_models dir. """

        _, max_acc = find_best_accuracy_from_saved_models(self.saved_models_dir)
        validation_accuracy = logs["val_binary_accuracy"]
        if validation_accuracy > max_acc:
            save_name = "model.e{0:03d}-acc{1:.4f}".format(
                epoch + 1, validation_accuracy)
            save_loc = os.path.join(self.saved_models_dir, save_name)
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                    "maximum TEST accuracy.\nSaving model at {location}")
                    .format(epoch + 1, validation_accuracy, location = save_loc))
            self.model.save(save_loc, save_format='tf')
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo model was "
                   "saved").format(epoch + 1, validation_accuracy))
        line = self.create_info_file_line(epoch, logs)
        update_file(self.info_save_file, line)
