from .utils import *

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
