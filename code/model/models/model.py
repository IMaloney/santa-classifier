import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, SpatialDropout2D, LocallyConnected2D
import model.hyperparameters as hp

class SantaClassifier(tf.keras.Model):
    def __init__(self, input=(hp.image_size, hp.image_size, 3)):
        super(SantaClassifier, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hp.min_learning_rate, momentum=hp.momentum)
        
        # self.l2_regularizer = tf.keras.regularizers.l2(l2=hp.l2)
        # , kernel_regularizer=self.l2_regularizer
        self.conv2d_1 = Conv2D(3, 3, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), input_shape=(hp.image_size, hp.image_size, 3))
        # self.pooling_1 = MaxPool2D()
        # self.dropout_1 = SpatialDropout2D(hp.dropout)
        # , kernel_regularizer=self.l2_regularizer
        # self.conv2d_2 = Conv2D(3, 4, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha))
        
        # self.dropout_1 = SpatialDropout2D(hp.dropout)
        # , kernel_regularizer=self.l2_regularizer
        # self.conv2d_3 = Conv2D(30, 5, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha))
        # self.pooling_2 = MaxPool2D()
        # self.conv2d_4 = Conv2D(100, 3, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha))
        # self.dropout_3 = SpatialDropout2D(hp.dropout)
        # self.pooling_3 = MaxPool2D()
        #  kernel_regularizer=self.l2_regularizer
        # self.conv2d_5 = Conv2D(250, 4, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha),)
        self.flatten = Flatten()
        # self.dense_1 = Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha))
        # self.dropout = Dropout(hp.dropout)
        # kernel_regularizer=self.l2_regularizer
        self.dense_2 = Dense(1, activation='sigmoid')
        

    def call(self, x):
        x = self.conv2d_1(x)
        # x = self.dropout_1(x)
        # x = self.pooling_1(x)
        # x = self.conv2d_2(x)
        # x = self.dropout_2(x)
        # x = self.conv2d_3(x)
        # x = self.pooling_2(x)
        # x = self.dropout_3(x)
        # x = self.conv2d_4(x)
        # x = self.pooling_3(x)
        # x = self.pooling_4(x)
        # x = self.conv2d_5(x)
        # x = self.conv2d_4(x)
        x = self.flatten(x)
        # x = self.dense_1(x)
        # x = self.dropout(x)
        x = self.dense_2(x)
        return x