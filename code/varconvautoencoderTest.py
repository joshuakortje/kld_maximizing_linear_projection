from gaussianprojection.utils import *
import gaussianprojection.algorithm1 as algo1
import gaussianprojection.algorithm2 as algo2
import gaussianprojection.lol as lol
import gaussianprojection.gradientdescent as gdesc
import gaussianprojection.autoencoder as autoencoder
import gaussianprojection.convautoencoder as cae
import gaussianprojection.varconvautoencoder as vae
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
import tensorflow as tf

print('Autoencoder with CIFAR-10')
training_p = 5000
test_p = 1000
r = 2
trim = 0
image_width = 32
vector_size = image_width*image_width*3
latent_dim = 2


cifar10_train_data, cifar10_test_data = get_cifar10_2_class(trim, training_p)
cifar10_train_data = reconstruct_image(cifar10_train_data, (2*training_p, image_width, image_width, 3))
cifar10_test_data = reconstruct_image(cifar10_test_data, (2*test_p, image_width, image_width, 3))

# Make and train the autoencoder
shape = (image_width, image_width, 3)
cifar_autoencoder = vae.VarConvAutoencoder(latent_dim, shape)
cifar_autoencoder.train_autoencoder(cifar10_train_data)

cifar_autoencoder.encoder.summary()
cifar_autoencoder.decoder.summary()

labels = {0: "Class 1",
          1: "Class 2"}
testing_labels = test_p * [0] + test_p * [1]

z_mean, _, _ = cifar_autoencoder.encoder.predict(cifar10_test_data)
plt.figure(figsize=(12, 10))
sc = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=testing_labels)
cbar = plt.colorbar(sc, ticks=range(2))
cbar.ax.set_yticklabels([labels.get(i) for i in range(2)])
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()
