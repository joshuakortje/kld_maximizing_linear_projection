import numpy as np
import pandas as pd
from .utils import *
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Define an autoencoder class for basic autoencoding
class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.size = int(np.prod(shape))
    #self.images = layers.Input(shape=shape)
    #self.vector_images = layers.Flatten()(self.images)
    # encoder
    #self.encoder_hidden = layers.Dense(self.size // 2, activation='sigmoid')(self.vector_images)
    #self.latent = layers.Dense(latent_dim, activation='sigmoid')(self.vector_images)
    # define decoder
    #self.decoder_hidden = layers.Dense(self.size // 2, activation='sigmoid')(self.latent)
    # output dense layer
    #self.decoder_output = layers.Dense(self.size, activation='sigmoid')(self.latent)
    #self.output_images = layers.Reshape(shape)(self.decoder_output)
    # define autoencoder model
    #self.autoencoder = Model(inputs=self.images, outputs=self.output_images)

    # define the encoder and decoder separately
    #self.encoder = Model(inputs=self.images, outputs=self.latent)
    #self.decoder = Model(inputs=self.latent, outputs=self.output_images)

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=shape),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(self.size, activation='sigmoid'),
      layers.Reshape(shape)
    ])

    self.compile(optimizer='adam', loss=losses.BinaryCrossentropy())

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  # Function to train the autoencoder
  def train_autoencoder(self, data_train, data_test):
    # Get the shape of the individual pieces of data for the autoencoder
    # We don't use it here (debug only), but this must match the shape passed into the autoencoder class
    data_shape = data_train.shape[1:]

    # Train the autoencoder
    history = self.fit(data_train, data_train,
                          epochs=5,
                          shuffle=True,
                          validation_data=(data_test, data_test))
