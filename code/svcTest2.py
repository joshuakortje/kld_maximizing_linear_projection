from gaussianprojection import *
from sklearn.model_selection import cross_val_score

#@title Testing the SVC with a set of synthetically generated data.
# Parameters
print('Classification test with Synthetic Data')
n = 6
training_p = 10000
r = 2
test_p = 1000
cv_factor = 5
attempts = 5
num_steps = 1000
large_mu = False # else Small Mu

# Generate the distributions and training data
# Data in the pickle file was generated with this function.
#mu, sigma, data = gen_gaussians_n_dim_samples(n, training_p, 1, 1)
if large_mu:
  file_path = '../../../../data/VisualizationExLargeMu.pickle'
else:
  file_path = '../../../../data/VisualizationExSmallMu.pickle'

with open(file_path, 'rb') as file:
  mu = pickle.load(file)
  sigma = pickle.load(file)

# Generate samples
data = gen_gaussian_samples(mu, sigma, training_p)

# Calculate the Ideal KLD (best case with perfect recovery)
original_kld = gaussian_kld((mu[0], sigma[0]), (mu[1], sigma[1]))
print('Best KLD')
print(original_kld.flatten()[0])

# Get the mean and covariance contributions to the KLD of the original distributions.
x_1 = [mu[0], sigma[0]]
x_2 = [mu[1], sigma[1]]
mu_kld = gaussian_kld_mu(x_1, x_2)
sigma_kld = gaussian_kld_sigma(x_1, x_2)
print('X Mean KLD Contribution: ' + str(mu_kld.flatten()[0]))
print('X Covariance KLD Contribution: ' + str(sigma_kld))

# Convert the training data to a single set of labeled data
# Using class 1 and class 2 for our labels
training_data = np.concatenate(data)
training_labels = training_p*[1] + training_p*[2]

# Generate test data for each class
class_1_data = np.random.multivariate_normal(mu[0].flatten(), sigma[0], test_p)
class_2_data = np.random.multivariate_normal(mu[1].flatten(), sigma[1], test_p)

# Define our pipelines
stage = 'dimensionReduction'
algorithm1_pipe = Pipeline([(stage, Algorithm1(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
algorithm2_pipe = Pipeline([(stage, Algorithm2(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
lol_pipe = Pipeline([(stage, Lol(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
gradient_ascent_pipe = Pipeline([(stage, GradientAscent(r=r, num_steps=num_steps, attempts=attempts, init_val=np.eye(r, n), verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
gradient_ascent_pipe[stage].name = 'Gradient Ascent (Algorithm 1)'
gradient_ascent_pipe2 = Pipeline([(stage, GradientAscent(r=r, num_steps=num_steps, attempts=attempts, init_val=np.eye(r, n), verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
gradient_ascent_pipe2[stage].name = 'Gradient Ascent (Algorithm 2)'
pipes = [algorithm1_pipe, gradient_ascent_pipe, algorithm2_pipe, gradient_ascent_pipe2, lol_pipe]
last_xform = None

for pipe in pipes:
    # If doing a Gradient ascent, use the last xform as the initial value
    if pipe[stage].name[:15] == 'Gradient Ascent':
        print('Found Gradient Ascent!')
        pipe[stage].reinit(r=r, init_val=last_xform)

    print("Testing with " + pipe[stage].name + "...")

    # Perform cross validation
    print("Performing Cross Validation...")
    pipe_score = cross_val_score(pipe, training_data, training_labels, cv=cv_factor)
    print(pipe[stage].name + " Cross-validation scores: ", pipe_score)
    print(pipe[stage].name + " Average cross-validation score: ", pipe_score.mean())

    # fit the approach
    pipe.fit(training_data, training_labels)

    # Get the Xforms (and whitening matrix if applicable) to use later
    pipe_xform = pipe[stage].xform
    pipe_whitening = pipe[stage].whitening
    last_xform = pipe[stage].full_xform

    # Get the QR decomposition
    Q_fact, R_fact = np.linalg.qr(pipe_xform.transpose())

    # Get the predictions for each class
    class_1_predictions = pipe.predict(class_1_data)
    class_2_predictions = pipe.predict(class_2_data)

    # Get the transformed data for each class
    # Apply whitening if applicable
    if pipe_whitening is None:
        class1_data_used = class_1_data
        class2_data_used = class_2_data
    else:
        class1_data_used = np.matmul(pipe_whitening[1], np.subtract(class_1_data, pipe_whitening[0]).transpose()).transpose()
        class2_data_used = np.matmul(pipe_whitening[1], np.subtract(class_2_data, pipe_whitening[0]).transpose()).transpose()
    reduced_data_class_1 = np.matmul(Q_fact.transpose(), class1_data_used.transpose())
    reduced_data_class_2 = np.matmul(Q_fact.transpose(), class2_data_used.transpose())

    # Plot the data on the scatter plot
    if r == 2:
        plt.figure()
        plt.scatter(reduced_data_class_1[0, :], reduced_data_class_1[1, :], marker="x", s=20, color='blue', label='Class 1')
        plt.scatter(reduced_data_class_2[0, :], reduced_data_class_2[1, :], s=20, facecolors='none', edgecolors='orange', label='Class 2')
        plt.legend()
        plt.axis('square')
        plt.title(pipe[stage].name + ' Projection to 2D')

    # Estimate the distribution parameters to calculate the new KLD
    q_mean1, q_cov1 = calculate_statistics(reduced_data_class_1.transpose())
    q_mean2, q_cov2 = calculate_statistics(reduced_data_class_2.transpose())
    new_kld = gaussian_kld((q_mean1, q_cov1), (q_mean2, q_cov2))[0]

    print(pipe[stage].name + ' KLD')
    print(new_kld)

    # Count up and report the correct classification percent
    total_count = len(class_1_predictions) + len(class_2_predictions)
    percent_correct = (list(class_1_predictions).count(1) + list(class_2_predictions).count(2)) / total_count
    print('Percent correct ' + pipe[stage].name + ': ' + f"{percent_correct: .2%}")

if r == 2:
  plt.show()
