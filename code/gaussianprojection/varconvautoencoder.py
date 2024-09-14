import numpy as np
import pandas as pd
from .utils import *
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Adapted from https://www.geeksforgeeks.org/variational-autoencoders/

# this sampling layer is the bottleneck layer of variational autoencoder,
# it uses the output from two dense layers z_mean and z_log_var as input,
# convert them into normal distribution and pass them to the decoder layer

class Sampling(layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VarConvAutoencoder(Model):
    def __init__(self, latent_dim, shape, **kwargs):
        super().__init__(**kwargs)

        # Define the encoder
        encoder_inputs = layers.Input(shape)
        x = layers.Conv2D(64, 3, activation="leaky_relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="leaky_relu")(x)
        mean = layers.Dense(latent_dim, name="mean")(x)
        log_var = layers.Dense(latent_dim, name="log_var")(x)
        z = Sampling()([mean, log_var])
        self.encoder = Model(encoder_inputs, [mean, log_var, z], name="encoder")
        self.encoder.summary()

        # Define the decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        y = layers.Dense(8 * 8 * 64, activation="leaky_relu")(latent_inputs)
        y = layers.Reshape((8, 8, 64))(y)
        y = layers.Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same")(y)
        y = layers.Conv2DTranspose(64, 3, activation="leaky_relu", strides=2, padding="same")(y)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(y)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

        # Define the metrics needed to guide the VAE training process
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

        self.compile(optimizer='adam')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Encoding and decoding
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Get the losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # Function to train the autoencoder
    def train_autoencoder(self, data_train):
        # Get the shape of the individual pieces of data for the autoencoder
        # We don't use it here (debug only), but this must match the shape passed into the autoencoder class
        data_shape = data_train.shape[1:]

        # Train the autoencoder
        history = self.fit(data_train, epochs=5, batch_size=128)
