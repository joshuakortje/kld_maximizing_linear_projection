from .utils import *

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
