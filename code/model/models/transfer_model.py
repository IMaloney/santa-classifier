import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, UpSampling2D, SpatialDropout2D, LocallyConnected2D
import model.hyperparameters as hp


class SantaClassifierExtension(tf.keras.Model):
    def __init__(self, input=(hp.image_size, hp.image_size, 3)):
        super(SantaClassifierExtension, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.min_learning_rate)
        self.l2_regularizer = tf.keras.regularizers.l2(l2=hp.l2)
        self.conv2d_1 = Conv2D(3, 4, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), input_shape=(hp.image_size, hp.image_size, 3), kernel_regularizer=self.l2_regularizer)
        self.upsampling_1 = UpSampling2D(10)
        self.conv2d_2 = Conv2D(5, 5, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        self.dropout_1 = SpatialDropout2D(hp.dropout)
        self.upsampling_2 = UpSampling2D(4)
        self.conv2d_3 = Conv2D(8, 5, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        # self.pooling_1 = MaxPool2D()
        self.dropout_2 = SpatialDropout2D(hp.dropout)        
        self.conv2d_4 = Conv2D(20, 4, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        # self.pooling_2 = MaxPool2D()
        # self.dropout_3 = SpatialDropout2D(hp.dropout)
        # self.conv2d_5 = Conv2D(32, 5, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        # self.pooling_3 = MaxPool2D()
        # self.dropout_4 = SpatialDropout2D(hp.dropout)
        # self.conv2d_6 = Conv2D(256, 2, activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        # self.pooling_4 = MaxPool2D()
        # self.dropout_5 = SpatialDropout2D(hp.dropout)
        self.flatten = Flatten()
        # self.dense_1 = Dense(512 , activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), kernel_regularizer=self.l2_regularizer)
        self.dropout = Dropout(hp.dropout)
        self.dense_2 = Dense(1, activation='sigmoid', kernel_regularizer=self.l2_regularizer)
        
    def call(self, x):
        x = self.conv2d_1(x)
        x = self.upsampling_1(x)
        x = self.conv2d_2(x)
        x = self.dropout_1(x)
        x = self.upsampling_2(x)
        x = self.conv2d_3(x)
        # x = self.pooling_1(x)
        x = self.dropout_2(x)
        x = self.conv2d_4(x)
        # x = self.pooling_2(x)
        # x = self.dropout_3(x)
        # x = self.conv2d_5(x)
        # x = self.pooling_3(x)
        # x = self.dropout_4(x)
        # x = self.conv2d_6(x)
        # x = self.pooling_4(x)
        # x = self.dropout_5(x)
        x = self.flatten(x)
        # x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x

class TransferSantaClassifier(tf.keras.Model):
    def __init__(self):
        super(TransferSantaClassifier, self).__init__()
        # tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(hp.image_size, hp.image_size, 3))
        self.base_model.trainable = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.min_learning_rate) # SGD(learning_rate=hp.min_learning_rate, momentum=hp.momentum, nesterov=True)
        self.santa_classifier = SantaClassifierExtension((6, 6, 2048))

    def call(self, x):
        x = self.base_model(x)
        x = self.santa_classifier(x)
        return x