import numpy as np
import pandas as pd
from .utils import *
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class ConvAutoencoder(Model):
    def __init__(self, shape):
        super(ConvAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=shape),
            layers.Conv2D(16, (3, 3), activation='leaky_relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='leaky_relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='leaky_relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='leaky_relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='softmax', padding='same')])

        self.compile(optimizer='adam', loss=losses.BinaryCrossentropy())

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # Function to train the autoencoder
    def train_autoencoder(self, data_train, data_test):
      # Get the shape of the individual pieces of data for the autoencoder
      # We don' use it here (debug only), but this must matche the shape passed into the autoencoder class
      data_shape = data_train.shape[1:]

      # Train the autoencoder
      history = self.fit(data_train, data_train,
                            epochs=5,
                            shuffle=True,
                            validation_data=(data_test, data_test))
