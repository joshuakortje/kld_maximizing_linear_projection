
#@title Importing Packages and Initializing Stuff
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
  def __init__(self, r):
    self.xform = None
    self.labels = None
    self.means = None
    self.covs = None
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
    self.means = defaultdict(list)
    self.covs = defaultdict(list)

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = sample_mean
      self.covs[self.labels[lab]] = sample_cov

    # Note: This part currently only works for 2 classes
    new_kld, self.xform = algorithm1_data(grouped_x, self.r)

    return self

  # Transform the data to the new subspace
  # X - data
  def transform(self, X):
    # Transform the data using the eigenvectors calculated in the fit step
    X = np.matmul(self.xform, X.transpose()).transpose()
    return X


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
  reduced_eigenvectors = np.array(eigenvectors[:r]).squeeze()

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
  def __init__(self, r):
    self.xform = None
    self.whitening = None
    self.labels = None
    self.means = None
    self.covs = None
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
    self.means = defaultdict(list)
    self.covs = defaultdict(list)

    # Get all the sample means/covs for each class
    for lab in range(len(self.labels)):
      sample_mean, sample_cov = calculate_statistics(grouped_x[lab])
      self.means[self.labels[lab]] = sample_mean
      self.covs[self.labels[lab]] = sample_cov

    # Note: This part only works for 2 classes
    new_kld, self.xform, self.whitening = algorithm2_data(grouped_x, self.r)

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
  def __init__(self, r):
    self.xform = None
    self.labels = None
    self.means = None
    self.covs = None
    self.r = r
    print('Initialising LoL transformer...')

  # Fit the data. That is, find the eigenvectors to reduce to the
  # new subspace
  # X - data
  # y - labels
  def fit(self, X, y):
    # Split the classes
    grouped_x, ordered_y = split_classes(X, y)

    self.labels = ordered_y
    self.means = defaultdict(list)
    self.covs = defaultdict(list)

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

