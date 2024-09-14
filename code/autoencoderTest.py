from gaussianprojection.utils import *
import gaussianprojection.algorithm1 as algo1
import gaussianprojection.algorithm2 as algo2
import gaussianprojection.lol as lol
import gaussianprojection.gradientdescent as gdesc
import gaussianprojection.autoencoder as autoencoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses

print('Autoencoder with CIFAR-10')
training_p = 5000
test_p = 1000
r = 2
trim = 0
image_width = 32
vector_size = image_width*image_width*3
latent_dim = 500

# Get 2 classes of CIFAR-10
cifar10_train_data, cifar10_test_data = get_cifar10_2_class(trim, training_p)
#(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# Make and train the autoencoder
shape = (vector_size,)
cifar_autoencoder = autoencoder.Autoencoder(latent_dim, shape)

cifar10_train_data = np.reshape(cifar10_train_data, (2*training_p, vector_size))
cifar10_test_data = np.reshape(cifar10_test_data, (2*test_p, vector_size))

cifar_autoencoder.train_autoencoder(cifar10_train_data, cifar10_test_data)

cifar_autoencoder.encoder.summary()
cifar_autoencoder.decoder.summary()

encoded_imgs = cifar_autoencoder.encoder(cifar10_test_data).numpy()
decoded_imgs = cifar_autoencoder.decoder(encoded_imgs).numpy()

test_images = reconstruct_image(cifar10_test_data, (2*test_p, image_width, image_width, 3))
decoded_images = reconstruct_image(decoded_imgs, (2*test_p, image_width, image_width, 3))

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(test_images[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_images[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

