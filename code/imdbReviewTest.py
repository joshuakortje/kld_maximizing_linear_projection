import numpy as np

from gaussianprojection.utils import *
import gaussianprojection.algorithm1 as algo1
import gaussianprojection.algorithm2 as algo2
import gaussianprojection.lol as lol
import gaussianprojection.svcPassthrough as svcpass
import gaussianprojection.gradientdescent as gdesc
import gaussianprojection.autoencoder as autoencoder
import gaussianprojection.convautoencoder as cae
import gaussianprojection.varconvautoencoder as vae
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

max_num_words = 10000  # after 10000 words there are only 66 instances over all reviews
max_words = 1000
training_p = 40000
test_p = 10000
r = 2

print('SVC with IMDB Review Sentiment')

(imdb_train_data, imdb_train_labels),  (imdb_test_data, imdb_test_labels) = get_imdb_reviews(max_num_words, training_p, max_words)

passthrough_dim = imdb_test_data.shape[1]

# Define our pipeline
stage = 'dimensionReduction'
cv_factor = 5
algorithm1_pipe = Pipeline([(stage, algo1.Algorithm1(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
algorithm2_pipe = Pipeline([(stage, algo2.Algorithm2(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
lol_pipe = Pipeline([(stage, lol.Lol(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
passthrough_pipe = Pipeline([(stage, svcpass.Passthrough(r=passthrough_dim, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
pipes = [algorithm1_pipe, algorithm2_pipe, lol_pipe]
last_xform = None


training_data = imdb_train_data
training_labels = imdb_train_labels + 1  # convert classes (0, 1) -> (1, 2)
class_1_data = imdb_test_data[np.where(imdb_test_labels == 0)]
class_2_data = imdb_test_data[np.where(imdb_test_labels == 1)]

for pipe in pipes:
    # If doing a Gradient ascent, use the last xform as the initial value
    if pipe[stage].name[:15] == 'Gradient Ascent':
        print('Found Gradient Ascent!')
        pipe[stage].reinit(r=r, init_val=last_xform)

    print("Testing with " + pipe[stage].name + "...")

    ## Perform cross validation
    #print("Performing Cross Validation...")
    #pipe_score = cross_val_score(pipe, training_data, training_labels, cv=cv_factor)
    #print(pipe[stage].name + " Cross-validation scores: ", pipe_score)
    #print(pipe[stage].name + " Average cross-validation score: ", pipe_score.mean())

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

