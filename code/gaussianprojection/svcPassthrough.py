import numpy as np

from .utils import *

# Define the a passthrough class to be compatible with the functions used elsewhere
# but not actually do any dimensin reduction.
class Passthrough(BaseEstimator, TransformerMixin):
  # Constructor
  # r - number of dimensions to reduce to
  def __init__(self, r, verbose):
    self.name = "Passthrough"
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
      print('Initialising Passthrough transformer...')
    self.r = r

  # Fit the data. That is, find the eigenvectors needed to transform
  # the data to the new r dimension subspace
  # X - data
  # y - labels
  def fit(self, X, y):
    # Define the transform to be the identity (no change)
    self.xform = np.identity(self.r)
    self.full_xform = np.identity(self.r)

    return self

  # Transform the data to the new subspace
  # X - data
  def transform(self, X):
    # no operation on the data
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
      print('Reinitialising Passthrough transformer...')
    self.r = r

  # Apply Algorithm 1 to a set of distribution parameters.
  def distribution_fit(self, mu, sigma):
    # return the kld based on the mu and sigma passed in since there is no change
    self.kld = gaussian_kld((mu[0], sigma[0]), (mu[1], sigma[1]))

    return self.kld

  # Transform the data to an orthogonal subspace that spans the projected subspace
  def transform_orthogonal(self, data):
    # Get the QR factorization of the transformation matrix
    Q_fact, R_fact = np.linalg.qr(self.xform)

    # Transform the data
    return np.matmul(Q_fact.transpose(), data.transpose())
