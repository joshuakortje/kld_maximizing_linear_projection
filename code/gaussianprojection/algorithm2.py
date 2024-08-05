from .utils import *

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
