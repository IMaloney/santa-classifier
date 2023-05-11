import numpy as np
import tensorflow as tf
import os
import model.hyperparameters as hp
from shared.utils import update_file

DEFAULT_LR_FOLDER = 'learning_rate'

# learning rate starts at min_learning_rate and increases smoothly to a maximum of max_learning_rate using a logistic function.
def scheduler_function(epoch):
    min_lr, max_lr = hp.min_learning_rate, hp.max_learning_rate
    return min_lr + 0.99 * (max_lr - min_lr) * (1 / (1 + np.exp(-0.1 * (epoch - 10))))

class LRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, lr_file):
        super(LRScheduler, self).__init__()
        self.lr_file = lr_file       
        self.schedule = scheduler_function
            
    def on_epoch_begin(self, epoch, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        line = f"Epoch: {epoch}\n\tLearning Rate: {lr}\n"
        update_file(self.lr_file, line)