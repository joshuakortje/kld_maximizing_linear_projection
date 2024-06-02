from gaussianprojection import *
from sklearn.model_selection import cross_val_score

#@title Testing the SVC with a set of synthetically generated data.
# Parameters
print('Classification test with Synthetic Data')
n = 6
training_p = 10000
r = 2
test_p = 1000
cv_factor = 2
large_mu = False # else Small Mu

# Generate the distributions and training data
# Data in the pickle file was generated with this function.
#mu, sigma, data = gen_gaussians_n_dim_samples(n, training_p, 1, 1)
if large_mu:
  file_path = 'VisualizationExLargeMu.pickle'
else:
  file_path = 'VisualizationExSmallMu.pickle'

with open(file_path, 'rb') as file:
  mu = pickle.load(file)
  sigma = pickle.load(file)

# Generate samples
data = gen_gaussian_samples(mu, sigma, training_p)

# Calculate the Ideal KLD (best case with perfect recovery)
original_kld = gaussian_kld((mu[0], sigma[0]), (mu[1], sigma[1]))

# Get the mean and covariance contributions to the KLD of the original distributions.
x_1 = [mu[0], sigma[0]]
x_2 = [mu[1], sigma[1]]
mu_kld = gaussian_kld_mu(x_1, x_2)
sigma_kld = gaussian_kld_sigma(x_1, x_2)
print('X Mean KLD Contribution: ' + str(mu_kld.flatten()[0]))
print('X Covariance KLD Contribution: ' + str(sigma_kld))

# Define our pipelines
algorithm1_pipe = Pipeline([('algorithm1', Algorithm1(r=r)),('scaler', StandardScaler()), ('svc', SVC())])
algorithm2_pipe = Pipeline([('algorithm2', Algorithm2(r=r)),('scaler', StandardScaler()), ('svc', SVC())])
lol_pipe = Pipeline([('lol', Lol(r=r)),('scaler', StandardScaler()), ('svc', SVC())])

# Convert the training data to a single set of labeled data
# Using class 1 and class 2 for our labels
training_data = np.concatenate(data)
training_labels = training_p*[1] + training_p*[2]

# Perform cross validation
print("Performing Cross Validation...")
algorithm1_score = cross_val_score(algorithm1_pipe, training_data, training_labels, cv=cv_factor)
algorithm2_score = cross_val_score(algorithm2_pipe, training_data, training_labels, cv=cv_factor)
lol_score = cross_val_score(lol_pipe, training_data, training_labels, cv=cv_factor)

print("Algorithm 1 Cross-validation scores:", algorithm1_score)
print("Algorithm 1 Average cross-validation score:", algorithm1_score.mean())
print("Algorithm 2 Cross-validation scores:", algorithm2_score)
print("Algorithm 2 Average cross-validation score:", algorithm2_score.mean())
print("Lol Cross-validation scores:", lol_score)
print("Lol Average cross-validation score:", lol_score.mean())

# fit both approaches
algorithm1_pipe.fit(training_data, training_labels)
algorithm2_pipe.fit(training_data, training_labels)
lol_pipe.fit(training_data, training_labels)

# Get the Xforms to use later
algorithm1_xform = algorithm1_pipe['algorithm1'].xform
algorithm2_xform = algorithm2_pipe['algorithm2'].xform
algorithm2_whitening = algorithm2_pipe['algorithm2'].whitening
lol_xform = lol_pipe['lol'].xform

# Get the QR decomposition to use later
Q1, R1 = np.linalg.qr(algorithm1_xform.transpose())
Q2, R2 = np.linalg.qr(algorithm2_xform.transpose())
Q3, R3 = np.linalg.qr(lol_xform.transpose())

# Generate test data for each class
class_1_data = np.random.multivariate_normal(mu[0].flatten(), sigma[0], test_p)
class_2_data = np.random.multivariate_normal(mu[1].flatten(), sigma[1], test_p)

# Get the predictions for each class
class_1_predictions_algo_1 = algorithm1_pipe.predict(class_1_data)
class_2_predictions_algo_1 = algorithm1_pipe.predict(class_2_data)
class_1_predictions_algo_2 = algorithm2_pipe.predict(class_1_data)
class_2_predictions_algo_2 = algorithm2_pipe.predict(class_2_data)
class_1_predictions_lol = lol_pipe.predict(class_1_data)
class_2_predictions_lol = lol_pipe.predict(class_2_data)

# Get the transformed data for each class
reduced_data_class_1_algo_1 = np.matmul(Q1.transpose(), class_1_data.transpose())
reduced_data_class_2_algo_1 = np.matmul(Q1.transpose(), class_2_data.transpose())
class1_data_whitened = np.matmul(algorithm2_whitening[1], np.subtract(class_1_data, algorithm2_whitening[0]).transpose())
class2_data_whitened = np.matmul(algorithm2_whitening[1], np.subtract(class_2_data, algorithm2_whitening[0]).transpose())
reduced_data_class_1_algo_2 = np.matmul(Q2.transpose(), class1_data_whitened)
reduced_data_class_2_algo_2 = np.matmul(Q2.transpose(), class2_data_whitened)
reduced_data_class_1_lol = np.matmul(Q3.transpose(), class_1_data.transpose())
reduced_data_class_2_lol = np.matmul(Q3.transpose(), class_2_data.transpose())

# Plot the data on the scatter plot
if r == 2:
  plt.figure()
  plt.scatter(reduced_data_class_1_algo_1[0,:], reduced_data_class_1_algo_1[1,:], marker="x", s=20, color='blue', label='Class 1')
  plt.scatter(reduced_data_class_2_algo_1[0,:], reduced_data_class_2_algo_1[1,:], s=20, facecolors='none', edgecolors='orange', label='Class 2')
  plt.legend()
  plt.axis('square')
  plt.title('Algorithm 1 Projection to 2D')

  plt.figure()
  plt.scatter(reduced_data_class_1_algo_2[0,:], reduced_data_class_1_algo_2[1,:], marker="x", s=20, color='blue', label='Class 1')
  plt.scatter(reduced_data_class_2_algo_2[0,:], reduced_data_class_2_algo_2[1,:], s=20, facecolors='none', edgecolors='orange', label='Class 2')
  plt.legend()
  plt.axis('square')
  plt.title('Algorithm 2 Projection to 2D')

  plt.figure()
  plt.scatter(reduced_data_class_1_lol[0, :], reduced_data_class_1_lol[1, :], marker="x", s=20, color='blue', label='Class 1')
  plt.scatter(reduced_data_class_2_lol[0, :], reduced_data_class_2_lol[1, :], s=30, facecolors='none', edgecolors='orange', label='Class 2')
  plt.legend()
  plt.axis('square')
  plt.title('LoL Projection to 2D')

# Estimate the distribution parameters to calculate the new KLD
q1_mean1, q1_cov1 = calculate_statistics(reduced_data_class_1_algo_1.transpose())
q1_mean2, q1_cov2 = calculate_statistics(reduced_data_class_2_algo_1.transpose())
q2_mean1, q2_cov1 = calculate_statistics(reduced_data_class_1_algo_2.transpose())
q2_mean2, q2_cov2 = calculate_statistics(reduced_data_class_2_algo_2.transpose())
q3_mean1, q3_cov1 = calculate_statistics(reduced_data_class_1_lol.transpose())
q3_mean2, q3_cov2 = calculate_statistics(reduced_data_class_2_lol.transpose())

algorithm1_kld = gaussian_kld((q1_mean1, q1_cov1), (q1_mean2, q1_cov2))
algorithm2_kld = gaussian_kld((q2_mean1, q2_cov1), (q2_mean2, q2_cov2))
lol_kld = gaussian_kld((q3_mean1, q3_cov1), (q3_mean2, q3_cov2))

# Print results for KLD
print('Best KLD')
print(original_kld.flatten()[0])
print('Algorithm 1 KLD')
print(algorithm1_kld[0])
print('Algorithm 2 KLD')
print(algorithm2_kld[0])
print('LoL KLD')
print(lol_kld[0])

# Count up and report the correct classification percent
total_count = len(class_1_predictions_algo_1) + len(class_2_predictions_algo_1)
percent_correct_algo_1 = (list(class_1_predictions_algo_1).count(1) + list(class_2_predictions_algo_1).count(2))/total_count
percent_correct_algo_2 = (list(class_1_predictions_algo_2).count(1) + list(class_2_predictions_algo_2).count(2))/total_count
percent_correct_lol = (list(class_1_predictions_lol).count(1) + list(class_2_predictions_lol).count(2))/total_count

print('Percent correct Algorithm 1: ' + f"{percent_correct_algo_1: .2%}")
print("Percent correct Algorithm 2: " + f"{percent_correct_algo_2: .2%}")
print('Percent correct LoL: ' + f"{percent_correct_lol: .2%}")

if r == 2:
  plt.show()
