import os

#@title Importing Packages and Initializing Stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import operator
import statistics
import copy
import pickle
import scipy.io as sio
from IPython.core.debugger import Pdb
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow.compat.v1 as tf
#import tensorflow as tf
#from tensorflow.python.keras.optimizers import adam_v2
from gaussianprojection import *

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


#@title Algorithm 1
# Function to determine the optimal subspace via Algorithm 1.
# This is done by taking the one dimensional subspace \Sigma_2^-1(\mu_2-\mu_1) and
# concatenating it with r-1 other one dimensional subspaces that maximize the
# KLD based on the Generalized Eigenvalues of the covariance matrices.
def algorithm1(mu, sigma, r):
  # sub_1_eig_vect = sigma1*(mu_1 - mu_2)^T
  mu_diff = np.subtract(mu[1], mu[0])
  mu_diff_normalized = mu_diff/np.linalg.norm(mu_diff)  # normalize since all the other vectors will be normalized
  sub_1_eig_vect = np.transpose(scipy.linalg.solve(sigma[1], mu_diff_normalized, assume_a='pos', check_finite=False))

  # Now solve for the remaining r-1 subspaces assuming the zero mean case
  eigenvalues, eigenvectors = scipy.linalg.eig(sigma[0], b=sigma[1])
  eigenvalues = np.real(eigenvalues)  # should always be real for a PSD matrix
  eigenvectors_transposed = eigenvectors.transpose()

  # Measure the KLD associated with each eigenvalue
  kld_array = [0.5*(lambda_i-1-math.log(lambda_i)) for lambda_i in eigenvalues]

  # Sort the KLD, eigenvectors, and eigenvalues by the KLD
  kld_array, eigenvectors, eigenvalues = zip(*sorted(zip(kld_array, eigenvectors_transposed, eigenvalues), key=operator.itemgetter(0), reverse=True))

  # Get the first r-1 values (values with the highest KLD)
  reduced_eigenvectors = np.array(eigenvectors[:r-1]).squeeze()

  # Construct the new subspace from concatenating the two above approaches
  if len(reduced_eigenvectors) != 0:
    concat_vectors = np.insert(np.atleast_2d(reduced_eigenvectors), [0], sub_1_eig_vect, axis=0)
  else:
    # Handle the case where r == 1
    concat_vectors = sub_1_eig_vect

  concat_vectors_T = np.atleast_2d(concat_vectors).transpose()

  # Since the new eigenvectors for the subspace are not orthogonal, we need to
  # calculate the new H1 covariance matrix when projecting to this subspace.
  h1_cov = np.matmul(np.matmul(concat_vectors, sigma[0]), concat_vectors_T)
  h2_cov = np.matmul(np.matmul(concat_vectors, sigma[1]), concat_vectors_T)
  h1_mu = np.matmul(concat_vectors, mu[0]).flatten()
  h2_mu = np.matmul(concat_vectors, mu[1]).flatten()

  # Get the new KLD
  new_kld = gaussian_kld((h1_mu, h1_cov), (h2_mu, h2_cov))

  return new_kld, concat_vectors

# Algorithm 1 function to run Algorithm 1 on data. This function will take in data,
# calculate the corresponding sample mean and variance, then run Algorithm 1 using
# the calculated statistics.
def algorithm1_data(data, r):
  # Calculate the sample mean and variance
  sample_mean = list()
  sample_variance = list()
  sample_stats = [sample_mean, sample_variance]
  for data_class in data:
    for stat, calc_val in zip(sample_stats, calculate_statistics(data_class)):
      stat.append(calc_val)

  # Use Algorithm 1 using the statistical parameters
  new_kld, new_eigenvectors = algorithm1(sample_mean, sample_variance, r)

  return new_kld, new_eigenvectors

# Define the transformer for Algorithm 1 to use in sklearn pipelines
class Algorithm1(BaseEstimator, TransformerMixin):
  # Constructor
  # r - number of dimensions to reduce to
  def __init__(self, r, verbose):
    self.name = "Algorithm 1"
    self.verbose = verbose
    self.whitening = None
    self.xform = None
    self.full_xform = None
    self.labels = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    if self.verbose:
      print('Initialising Algorithm 1 transformer...')
    self.r = r

  # Fit the data. That is, find the eigenvectors needed to transform
  # the data to the new r dimension subspace
  # X - data
  # y - labels
  def fit(self, X, y):
    # Split the classes
    grouped_x, ordered_y = split_classes(X, y)

    self.labels = ordered_y

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = sample_mean
      self.covs[self.labels[lab]] = sample_cov

    # Note: This part currently only works for 2 classes
    new_kld, self.xform = algorithm1_data(grouped_x, self.r)
    self.full_xform = self.xform

    return self

  # Transform the data to the new subspace
  # X - data
  def transform(self, X):
    # Transform the data using the eigenvectors calculated in the fit step
    X = np.matmul(self.xform, X.transpose()).transpose()
    return X

  # Reinitializes the class with a new r value
  def reinit(self, r, init_val):
    self.xform = None
    self.labels = None
    self.whitening = None
    self.full_xform = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    if self.verbose:
      print('Reinitialising Algorithm 1 transformer...')
    self.r = r

  # Apply Algorithm 1 to a set of distribution parameters.
  def distribution_fit(self, mu, sigma):
    # Save off the mean and covariance
    for dist in range(len(mu)):
      self.means[dist] = mu[dist]
      self.covs[dist] = sigma[dist]

    # Perform Algorithm 1 operation on the distributions
    self.kld, self.xform = algorithm1(mu, sigma, self.r)
    self.full_xform = self.xform

    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Calculate the new distributions and the distributions in the orthogonal subspace
    for dist in range(len(mu)):
      self.new_means[dist] = np.matmul(self.xform, mu[dist])
      self.new_covs[dist] = np.matmul(self.xform, np.matmul(sigma[dist], self.xform.transpose()))
      self.orth_means[dist] = np.matmul(Q_fact.transpose(), mu[dist])
      self.orth_covs[dist] = np.matmul(Q_fact.transpose(), np.matmul(sigma[dist], Q_fact))

    return self.kld

  # Transform the data to an orthogonal subspace that spans the projected subspace
  def transform_orthogonal(self, data):
    # Get the QR factorization of the transformation matrix
    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Transform the data
    return np.matmul(Q_fact.transpose(), data.transpose())

#@title Algorithm 2

# Function to determine the optimal subspace via Algorithm 2.
# This is done by taking the eigendecomposition of the covariance matrix
# and then ordering the rank 1 subspaces by the KLD between N(u_i^T mu, lambda_i)
# and the standard normal distribution. Then simply take the r largest ones.
# This function assumes that the distribution has been whitened and takes in
# only the H2 distribution
# mu - mean
# sigma - covariance
# r - number of dimensions to reduce to
def algorithm2(mu, sigma, r):
  # Eigen decomposition
  # Should not encounter an imaginary components since
  # matrices are assumed to be PD
  eigenvalues, eigenvectors = np.linalg.eig(sigma)
  eigenvalues = np.real(eigenvalues)
  eigenvectors = np.real(eigenvectors)

  # Transpose so we can easily index to get each vector
  eigenvectors_transposed = eigenvectors.transpose()

  # Get the KLD for each 1-D subspace
  kld_array = [gaussian_kld((0, 1), (np.dot(u_i, mu.flatten()), lambda_i)) for u_i, lambda_i in zip(eigenvectors_transposed, eigenvalues)]

  # Sort the KLD, eigenvectors, and eigenvalues by the KLD values
  kld_array, eigenvectors, eigenvalues = zip(*sorted(zip(kld_array, eigenvectors_transposed, eigenvalues), key=lambda x: x[0], reverse=True))

  # Get the first r values (values with the highest KLD)
  reduced_kld = kld_array[:r]
  reduced_eigenvectors = np.atleast_2d(np.array(eigenvectors[:r]).squeeze())

  # Calculating the new sigma KLD
  new_kld = sum(reduced_kld)

  return new_kld, reduced_eigenvectors

# Algorithm 2 function to run Algorithm 2 on data. This function will take in data,
# calculate the corresponding sample mean and variance, then run Algorithm 2 using
# the calculated statistics.
# data - input data as 2 classes
# r - the number of dimensions to reduce to
def algorithm2_data(data, r):
  # Calculate the sample mean and variance
  sample_mean = list()
  sample_variance = list()
  sample_stats = [sample_mean, sample_variance]
  for data_class in data:
    for stat, calc_val in zip(sample_stats, calculate_statistics(data_class)):
      stat.append(calc_val)

  # Use Algorithm 2 using the statistical parameters
  # First we whiten and then pass the statistical parameters to the algorithm
  diag_mean, diag_var, whitening_params = diagonalize_h1(sample_mean, sample_variance)
  new_kld, new_eigenvectors = algorithm2(diag_mean[1], diag_var[1], r)

  return new_kld, new_eigenvectors, whitening_params

# Define the transformer for Algorithm 2 to use in sklearn pipelines
class Algorithm2(BaseEstimator, TransformerMixin):
  # Constructor
  # r - dimensions to reduce to
  def __init__(self, r, verbose):
    self.name = "Algorithm 2"
    self.verbose = verbose
    self.xform = None
    self.whitening = None
    self.full_xform = None
    self.labels = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    if self.verbose:
      print('Initialising Algorithm 2 transformer...')
    self.r = r

  # Fit the data. That is, find the eigenvectors to reduce to the
  # new subspace
  # X - data
  # y - labels
  def fit(self, X, y):
    # Split the classes
    grouped_x, ordered_y = split_classes(X, y)

    self.labels = ordered_y

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = sample_mean
      self.covs[self.labels[lab]] = sample_cov

    # Note: This part only works for 2 classes
    new_kld, self.xform, self.whitening = algorithm2_data(grouped_x, self.r)

    self.full_xform = np.matmul(self.xform, self.whitening[1])

    return self

  # Transform the data to the reduced dimension subspace
  # X - data
  def transform(self, X):
    # First we must whiten the matrix since the xform was calculated under the
    # assumption of a whitened distributions
    X_whitened = np.matmul(self.whitening[1], np.subtract(X, self.whitening[0]).transpose())
    # Transform the data using the eigenvectors calculated in the fit step
    X = np.matmul(self.xform, X_whitened).transpose()
    return X

  # Reinitializes the class with a new r value
  def reinit(self, r, init_val):
    self.xform = None
    self.whitening = None
    self.full_xform = None
    self.labels = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    if self.verbose:
      print('Reinitialising Algorithm 2 transformer...')
    self.r = r

  # Apply Algorithm 2 to a set of distribution parameters.
  def distribution_fit(self, mu, sigma):
    # Save off the mean and covariance
    for dist in range(len(mu)):
      self.means[dist] = mu[dist]
      self.covs[dist] = sigma[dist]

    # Perform Algorithm 2 operation on the distributions
    diag_mean, diag_var, self.whitening = diagonalize_h1(mu, sigma)
    self.kld, self.xform = algorithm2(diag_mean[1], diag_var[1], self.r)
    self.full_xform = np.matmul(self.xform, self.whitening[1])

    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Calculate the new distributions and the distributions in the orthogonal subspace
    for dist in range(len(diag_mean)):
      self.new_means[dist] = np.matmul(self.xform, diag_mean[dist])
      self.new_covs[dist] = np.matmul(self.xform, np.matmul(diag_var[dist], self.xform.transpose()))
      self.orth_means[dist] = np.matmul(Q_fact.transpose(), diag_mean[dist])
      self.orth_covs[dist] = np.matmul(Q_fact.transpose(), np.matmul(diag_var[dist], Q_fact))

    return self.kld

  # Transform the data to an orthogonal subspace that spans the projected subspace
  def transform_orthogonal(self, data):
    # Get the QR factorization of the transformation matrix
    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Transform the data
    return np.matmul(Q_fact.transpose(), data.transpose())

#@title LOL

# This function implements LOL with r = 2 on two distributions for which the
# covariance matrices are not the same.
# This is done by calculating the covariance matrix of all samples
# (both classes) and taking the largest principal component and concatenating that
# with mu_2-mu_1.
# data - data from each class
# r - dimension to reduce to
def adapted_lol_2d(data, r):
  p1_data = data[0]
  p2_data = data[1]

  # Get the normalized mean difference
  mu_1, cov_1 = calculate_statistics(p1_data)
  mu_2, cov_2 = calculate_statistics(p2_data)
  mean_diff = np.subtract(mu_2, mu_1)
  mean_diff_normalized = mean_diff/np.linalg.norm(mean_diff)

  # Get the covariance of all the samples (both classes)
  all_samples = np.concatenate(data)
  total_mean, total_cov = calculate_statistics(all_samples)

  # Eigen decomposition to get the principal component
  eigenvalues, eigenvectors = np.linalg.eig(total_cov)
  eigenvectors_transposed = eigenvectors.transpose()

  # Sort the eigenvectors and eigenvalues by the magnitude of the eigenvalues to order the principal components
  eigenvalues, eigenvectors_transposed = zip(*sorted(zip(eigenvalues, eigenvectors_transposed), key=operator.itemgetter(0), reverse=True))
  larget_prin_comp = eigenvectors_transposed[:r-1]

  # Construct the new subspace from concatenating the two above approaches
  if len(larget_prin_comp) != 0:
    concat_vectors = np.insert(np.atleast_2d(larget_prin_comp), [0], mean_diff_normalized, axis=0)
  else:
    # Handle the case where r == 1
    concat_vectors = mean_diff_normalized

  return concat_vectors

class Lol(BaseEstimator, TransformerMixin):
  # Constructor
  # r - number of dimensions to reduce to
  def __init__(self, r, verbose):
    self.name = "LoL"
    self.verbose = verbose
    self.whitening = None
    self.full_xform = None
    self.xform = None
    self.labels = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.r = r
    if self.verbose:
      print('Initialising LoL transformer...')

  # Fit the data. That is, find the eigenvectors to reduce to the
  # new subspace
  # X - data
  # y - labels
  def fit(self, X, y):
    # Split the classes
    grouped_x, ordered_y = split_classes(X, y)

    self.labels = ordered_y

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = sample_mean
      self.covs[self.labels[lab]] = sample_cov

    # Note: This part currently only works for 2 classes
    self.xform = adapted_lol_2d(grouped_x, self.r)

    return self

  # Transform the data to the reduced dimension subspace
  # X - data
  def transform(self, X):
    # Transform the data using the eigenvectors calculated in the fit step
    X = np.matmul(self.xform, X.transpose()).transpose()
    return X

  # Transform the data to an orthogonal subspace that spans the projected subspace
  def transform_orthogonal(self, data):
    # Get the QR factorization of the transformation matrix
    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Transform the data
    return np.matmul(Q_fact.transpose(), data.transpose())

#@title Gradient Ascent Code

# Implimentation of Gradient Ascent algorithm to numerically solve for the local minimum for the KLD
def gradient_ascent(mu, sigma, num_steps, r, init_var, verbose):
  # Make a variable for the input of the function
  A = tf.Variable(init_var, trainable=True, dtype=tf.float64, name='A')

  def kld_objective():
    # Calculate the necessary determinants and inverses
    projected_mu_1 = tf.linalg.matmul(A, mu[0])
    projected_mu_2 = tf.linalg.matmul(A, mu[1])
    projected_sigma_1 = tf.linalg.matmul(A, tf.linalg.matmul(sigma[0], tf.linalg.matrix_transpose(A)))
    projected_sigma_2 = tf.linalg.matmul(A, tf.linalg.matmul(sigma[1], tf.linalg.matrix_transpose(A)))
    sigma_1_det = tf.linalg.det(projected_sigma_1)
    sigma_2_det = tf.linalg.det(projected_sigma_2)
    sigma_2_inv = tf.linalg.inv(projected_sigma_2)
    # KLD term is negated since we are minimizing the objective function
    # and maximizing the KLD
    kld_term = -0.5 * (tf.math.log(sigma_2_det / sigma_1_det) - r + tf.linalg.trace(
      tf.linalg.matmul(sigma_2_inv, projected_sigma_1)) +
                       tf.linalg.matmul(tf.linalg.matmul(
                         tf.linalg.matrix_transpose(tf.math.subtract(projected_mu_2, projected_mu_1)), sigma_2_inv),
                                        tf.math.subtract(projected_mu_2, projected_mu_1)))
    return kld_term

  #optimizer = tf.train.GradientDescentOptimizer(0.01)
  #initial_learning_rate = 0.001
  #lr_schedule = adam_v2.optimizer_v2.learning_rate_schedule.ExponentialDecay(initial_learning_rate,decay_steps=100000,decay_rate=0.96, staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
  kld_collection = list()
  gradient_collection = list()
  for step in range(num_steps):
    if verbose:
      print("step", step, "KLD:", kld_objective().numpy())
    kld_collection.append(kld_objective().numpy())
    new_gradient = optimizer.compute_gradients(kld_objective, var_list=[A])[0][0].numpy()
    gradient_collection.append(new_gradient)
    optimizer.minimize(kld_objective, var_list=[A])
    #print("Learning Rate: " + str(optimizer._lr))
    #if step > 100:
    #  if abs(kld_collection[step] - kld_collection[step-100]) < 10 and max(abs(np.gradient(np.array(kld_collection[step-100:step]).flatten())[1:-2])) < 10:
    #    if verbose:
    #      print('Breaking early at step ' + str(step))
    #    break
    #if verbose:
    #  print("Norm: " + str(np.linalg.norm(new_gradient)))

  return A.numpy(), -1.0 * kld_objective().numpy(), gradient_collection

# Gradient Ascent class for Pipelines
class GradientAscent(BaseEstimator, TransformerMixin):
  # Constructor
  # r - number of dimensions to reduce to
  # init_val - the value to initialize the gradient ascent algorithm with
  def __init__(self, r, num_steps, attempts, init_val, verbose):
    self.name = "Gradient Ascent"
    self.verbose = verbose
    self.whitening = None
    self.xform = None
    self.full_xform = None
    self.labels = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    self.init_val = init_val
    self.num_steps = num_steps
    self.attempts = attempts
    if self.verbose:
      print('Initialising Gradient Ascent transformer...')
    self.r = r

  def fit(self, X, y):
    # Split the classes
    grouped_x, ordered_y = split_classes(X, y)

    self.labels = ordered_y

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = np.atleast_2d(sample_mean).transpose()
      self.covs[self.labels[lab]] = np.atleast_2d(sample_cov)

    # Note: This part only works for 2 classes
    if self.init_val is None:
      init_arg = np.eye(self.r, X.shape[1])
    else:
      init_arg = self.init_val
    sorted_labels = sorted(list(set(self.labels)))
    sorted_means = [self.means.get(k) for k in sorted_labels]
    sorted_covs = [self.covs.get(k) for k in sorted_labels]
    xform_list = list()
    kld_list = list()
    for attempt in range(self.attempts):
      # Make a new random matrix each time
      rand_matrix = generate_rand_mat(self.r, sorted_means[0].shape[0]).transpose()
      temp_xform, grad_ascent_kld, grad_ascent_grads = gradient_ascent(sorted_means, sorted_covs, self.num_steps, self.r, init_arg, self.verbose)
      xform_list.append(temp_xform)
      kld_list.append(grad_ascent_kld.flatten()[0])

    index_max = np.argmax(kld_list)
    self.xform = xform_list[index_max]
    self.kld = kld_list[index_max]

    return self

  # Transform the data to the reduced dimension subspace
  # X - data
  def transform(self, X):
    # Transform the data using the eigenvectors calculated in the fit step
    X = np.matmul(self.xform, X.transpose()).transpose()
    return X

  # Reinitializes the class with a new r value
  def reinit(self, r, num_steps=None, attempts=None, init_val=None):
    self.xform = None
    self.labels = None
    self.whitening = None
    self.full_xform = None
    self.means = defaultdict(list)
    self.covs = defaultdict(list)
    self.kld = None
    self.new_means = defaultdict(list)
    self.new_covs = defaultdict(list)
    self.orth_means = defaultdict(list)
    self.orth_covs = defaultdict(list)
    self.init_val = init_val
    if num_steps is not None:
      self.num_steps = num_steps
    if attempts is not None:
      self.attempts = attempts
    if self.verbose:
      print('Reinitialising Gradient Ascent transformer...')
    self.r = r

  # Apply Gradient Ascent to a set of distribution parameters.
  def distribution_fit(self, mu, sigma):
    # Save off the mean and covariance
    for dist in range(len(mu)):
      self.means[dist] = mu[dist]
      self.covs[dist] = sigma[dist]

    # Perform Gradient Ascent operation on the distributions
    if self.init_val is None:
      init_arg = np.eye(self.r, mu.shape[1])
    else:
      init_arg = self.init_val

    xform_list = list()
    kld_list = list()
    for attempt in range(self.attempts):
      # Make a new random matrix each time
      rand_matrix = generate_rand_mat(self.r, mu.shape[1]).transpose()
      temp_xform, grad_ascent_kld, grad_ascent_grads = gradient_ascent(mu, sigma, self.num_steps, self.r, init_arg, self.verbose)
      xform_list.append(temp_xform)
      kld_list.append(grad_ascent_kld.flatten()[0])

    index_max = np.argmax(kld_list)
    self.xform = xform_list[index_max]
    self.kld = kld_list[index_max]

    Q_fact, R_fact = np.linalg.qr(self.xform.transpose())

    # Calculate the new distributions and the distributions in the orthogonal subspace
    for dist in range(len(mu)):
      self.new_means[dist] = np.matmul(self.xform, mu[dist])
      self.new_covs[dist] = np.matmul(self.xform, np.matmul(sigma[dist], self.xform.transpose()))
      self.orth_means[dist] = np.matmul(Q_fact.transpose(), mu[dist])
      self.orth_covs[dist] = np.matmul(Q_fact.transpose(), np.matmul(sigma[dist], Q_fact))

    return self.kld

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

