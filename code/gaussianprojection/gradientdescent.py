from .utils import *

#@title Gradient Descent Code

# Implimentation of Gradient Descent algorithm to numerically solve for the local minimum for the KLD
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
