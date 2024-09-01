#@title Importing Packages and Initializing Stuff
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import scipy
import math
import operator
import statistics
import copy
from keras.datasets import cifar10
import scipy.io as sio
from IPython.core.debugger import Pdb
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow.compat.v1 as tf

# Use SVD to create a dxr matrix of rank r
# H = UPV where
# U is dxd and invertible
# P is a structure where the top rxr portion is
# identity and then the bottom d-r rows are all zero
# V is rxr and invertible
def generate_rand_mat(r, d):
    # Create a dxd invertible matrix
    U = np.random.rand(d, d)
    u_x = np.sum(np.abs(U), axis=1)
    np.fill_diagonal(U, u_x)

    # Create a rxr invertible matrix
    V = np.random.rand(r, r)
    v_x = np.sum(np.abs(V), axis=1)
    np.fill_diagonal(V, v_x)

    # Create P which is rank r
    P = np.identity(r)
    P_app = np.zeros([d-r, r])
    P = np.vstack((P, P_app))
    #print("P")
    #print(P)

    H = np.matmul(U, np.matmul(P, V))
    H = np.random.randn(d, r)
    #print("H")
    #print(H)
    return H

# Helper function to split classes by their labels
# This functin will return the data and labels grouped by class
# X - data list
# y - label list corresponding to the data
def split_classes(X, y):
  data_classes = list()
  ordered_labels = list(set(y))

  for cls in ordered_labels:
    temp_class = list()
    for point in range(len(y)):
      if y[point] == cls:
        temp_class.append(X[point])
    data_classes.append(np.array(temp_class))

  return np.array(data_classes), ordered_labels


#@title KLD Calculation for Gaussians

# Function to calculate the mean contributions to
# the KLD for Gaussian distributions
# p_1~N(mu_1, sigma_1)
# p_2~N(mu_2, sigma_2)
def gaussian_kld_mu(p_1, p_2):
  mu_1 = p_1[0]
  mu_2 = p_2[0]
  sigma_2 = p_2[1]

  if isinstance(mu_1, int):
    # Special handling for the n = 1 case
    kld = 0.5 * (mu_2 - mu_1) * (1 / sigma_2) * (mu_2 - mu_1)
  else:
    n = len(mu_1)  # should be the same whether we use mu_1, mu_2, sigma_1, or sigma_2
    # Calculate the necessary inverse
    sigma_2_inv = np.linalg.inv(sigma_2)

    # KLD = 0.5 * (\mu-2 - \mu_1)^T(\Sigma_2^-1)(\mu-2 - \mu_1)
    kld = 0.5 * np.matmul(np.matmul(np.subtract(mu_2, mu_1).reshape(1, -1), sigma_2_inv), np.subtract(mu_2, mu_1))

  return kld

# Function to calculate the covariance
# contribution to the KLD for Gaussian distributions
# p_1~N(mu_1, sigma_1)
# p_2~N(mu_2, sigma_2)
def gaussian_kld_sigma(p_1, p_2):
  mu_1 = p_1[0]
  sigma_1 = p_1[1]
  sigma_2 = p_2[1]

  if isinstance(mu_1, int):
    # Special handling for the n = 1 case
    kld = 0.5*(math.log(sigma_2/sigma_1) - 1 + sigma_1/sigma_2)
  else:
    n = len(mu_1) # should be the same whether we use mu_1, mu_2, sigma_1, or sigma_2
    # Calculate the necessary determinants and inverses
    # Calculate in logspace to avoid overflow. Since the matrices are PD we
    # know the sign is always positive. Then we can subtract the exponents to
    # get the logarithm component of the KLD.
    sigma_1_det_log = np.linalg.slogdet(sigma_1)[1]
    sigma_2_det_log = np.linalg.slogdet(sigma_2)[1]
    sigma_log_component = sigma_2_det_log-sigma_1_det_log
    sigma_2_inv = np.linalg.inv(sigma_2)

    # KLD = 0.5(log(|sigma2|/|sigma1|) - n + Tr(sigma2^-1 sigma1))
    kld = 0.5*(sigma_log_component - n + np.matrix.trace(np.matmul(sigma_2_inv, sigma_1)))

  return kld

# Function to calculate the KLD for Gaussian distributions
# p_1~N(mu_1, sigma_1)
# p_2~N(mu_2, sigma_2)
def gaussian_kld(p_1, p_2):
  mu_1 = p_1[0]
  sigma_1 = p_1[1]
  mu_2 = p_2[0]
  sigma_2 = p_2[1]

  if isinstance(mu_1, int):
    # Special handling for the n = 1 case
    kld = 0.5*(math.log(sigma_2/sigma_1) - 1 + sigma_1/sigma_2 + (mu_2 - mu_1)*(1/sigma_2)*(mu_2-mu_1))
  else:
    n = len(mu_1) # should be the same whether we use mu_1, mu_2, sigma_1, or sigma_2
    # Calculate the necessary determinants and inverses
    sigma_1_det = np.linalg.det(sigma_1)
    sigma_2_det = np.linalg.det(sigma_2)
    sigma_2_inv = np.linalg.inv(sigma_2)

    # KLD = 0.5(log(|sigma2|/|sigma1|) - n + Tr(sigma2^-1 sigma1) + (\mu-2 - \mu_1)^T(\Sigma_2^-1)(\mu-2 - \mu_1))
    kld = 0.5*(math.log(sigma_2_det/sigma_1_det) - n + np.matrix.trace(np.matmul(sigma_2_inv, sigma_1)) +
              np.matmul(np.matmul(np.subtract(mu_2, mu_1).reshape(1,-1), sigma_2_inv), np.subtract(mu_2, mu_1)))

  return kld

#@title Calculate Data Statistics

# This function will take in a set of data can calculate the sample mean
# and sample variance
# data - each row is an n-dimension sample
def calculate_statistics(data):
  # Mean for each dimension of the samples
  sample_mean = np.mean(data, axis=0)

  # Need to transpose since numpy expects each row to represent the same variable
  sample_variance = np.cov(data.transpose())

  return sample_mean, sample_variance

#@title Diagonalize H1 Distribution

# Function to diagonalize the H1 distribution. Take
# p_1~N(mu_1, sigma_1)
# p_2~N(mu_2, sigma_2)
# and convert to
# q_1~N(0, I)
# q_2~N(mu, sigma)
# Returns mean vectors, covariance matrices,
def diagonalize_h1(mu, sigma):
  mu_1 = mu[0]
  sigma_1 = sigma[0]
  mu_2 = mu[1]
  sigma_2 = sigma[1]

  # The q_1 distribution is just standard normal
  q_mu_1 = np.zeros(len(mu_1))
  q_mu_1 = q_mu_1.reshape(-1,1)
  q_sigma_1 = np.identity(len(sigma_1))

  # The q_2 distribution calculations
  # Imaginary components should all be 0 because the matrix is PD.
  # mu = (sigma_1)^(-1/2)(mu_2 - mu_1)
  # sigma = (sigma_1)^(-1/2)sigma_2(sigma_1)^(-1/2)
  sigma_1_sqrt_inv = np.real(np.linalg.inv(scipy.linalg.sqrtm(sigma_1)))
  q_mu_2 = np.matmul(sigma_1_sqrt_inv, np.subtract(mu_2, mu_1))
  q_sigma_2 = np.matmul(sigma_1_sqrt_inv, np.matmul(sigma_2, sigma_1_sqrt_inv))

  means = [q_mu_1, q_mu_2]
  covs = [q_sigma_1, q_sigma_2]
  whitening_params = [mu_1, sigma_1_sqrt_inv]
  return means, covs, whitening_params

#@title Functions to Generate Distributions for 2 classes of Gaussians

# This function will set up 2 random Gaussian distributions with n dimensions.
# The function will also allow bringing the distributions as close as you want
# by setting a and b close to 0 to make the covariance and or mean of the distributions the same
# n - number of dimensions
# a - linear coeff on the mean
# b - linear coeff on the covariance
def gen_gaussian_n_dim(n, a=1, b=1):

  # p_1
  rand_matrix_1 = np.random.rand(n, n)
  cov_1 = np.dot(rand_matrix_1, rand_matrix_1.transpose())
  mu_1 = np.random.randn(n, 1)

  # p_2
  rand_matrix_2 = np.random.rand(n, n)
  cov_2 = np.dot(rand_matrix_2, rand_matrix_2.transpose())
  mu_2 = np.random.randn(n, 1)

  # Weight the class 1 mean and covariance as a linear combination of the two above distributions
  # This allows us to create distributions arbitrarily close to each other if desired
  mu_1 = np.add(a*mu_1, (1-a)*mu_2)
  cov_1 = np.add(b*cov_1, (1-b)*cov_2)

  means = [mu_1, mu_2]
  covs = [cov_1, cov_2]
  return means, covs

# This function will generate data samples to go along with the gaussian
# distribution parameters passed in
# mu - list of the means
# sigma - list of the covariances
# p - number of samples to generate
def gen_gaussian_samples(mu, sigma, p):
  # Generate the data for each class
  x_1 = np.random.multivariate_normal(mu[0].flatten(), sigma[0], p)
  x_2 = np.random.multivariate_normal(mu[1].flatten(), sigma[1], p)
  data = [x_1, x_2]

  return data


# This function will set up 2 random Gaussian distributions with n dimensions
# and pass back both the distribution parameters and samples from the distributions.
# The function will also allow bringing the distributions as close as you want
# by setting a and b close to 0 to make the covariance and or mean of the distributions the same
# n - the number of dimensions in the distributions
# p - the number of samples to generate
# a - linear coeff on the mean
# b - linear coeff on the covariance
def gen_gaussians_n_dim_samples(n, p, a, b):
  # Create the distribution randomly
  mu, sigma = gen_gaussian_n_dim(n, a, b)

  # Generate the data
  data = gen_gaussian_samples(mu, sigma, p)

  return mu, sigma, data


def remove_edges(data, dim1, dim2):
  print()

#@title Read in the CIFAR-10 dataset

def get_cifar10_2_class(trim, train_size):
  training_data_size = train_size*10
  total_data_size = 5000*10
  test_data_size = 10000
  width = 32
  new_width = width - trim*2
  dim_size = new_width*new_width*3
  (train_x, train_y), (test_x, test_y) = cifar10.load_data()
  #train_x = np.tan(train_x)
  #test_x = np.tan(test_x)
  # Option to get rid of the edges of each picture
  if trim > 0:
    train_x = np.delete(train_x, np.add(np.arange(trim), width - trim).tolist(), 1)
    train_x = np.delete(train_x, np.add(np.arange(trim), width - trim).tolist(), 2)
    train_x = np.delete(train_x, np.arange(trim).tolist(), 1)
    train_x = np.delete(train_x, np.arange(trim).tolist(), 2)
    test_x = np.delete(test_x, np.add(np.arange(trim), width - trim).tolist(), 1)
    test_x = np.delete(test_x, np.add(np.arange(trim), width - trim).tolist(), 2)
    test_x = np.delete(test_x, np.arange(trim).tolist(), 1)
    test_x = np.delete(test_x, np.arange(trim).tolist(), 2)
  train_x = train_x.reshape(total_data_size, dim_size)
  test_x = test_x.reshape(test_data_size, dim_size)

  train_data, train_labels = split_classes(train_x, train_y.flatten().tolist())
  test_data, test_classes = split_classes(test_x, test_y.flatten().tolist())

  train_data_trunc = train_data[:, :train_size, :]
  test_data_trunc = test_data[:, :train_size, :]

  # Get cat and dog classes
  cifar10_cat_dog_training_data = [train_data_trunc[1], train_data_trunc[9]]
  cifar10_cat_dog_test_data = [test_data_trunc[1], test_data_trunc[9]]
  return cifar10_cat_dog_training_data, cifar10_cat_dog_test_data

