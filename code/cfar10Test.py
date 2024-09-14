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

#@title SVC with CIFAR-10
print('SVC with CIFAR-10')
training_p = 5000
sub_train_set = int((4/5)*training_p)
test_p = 1000
r = 2
trim = 0
image_width = 32
vector_size = image_width*image_width*3
latent_dim = 500
use_autoencoder = True
type_autoencoders = ('basic', 'convolutional', 'variational')
type_autoencoder = type_autoencoders[2]
if type_autoencoder == 'convolutional':
    latent_dim = 512

# Get 2 classes of CIFAR-10 (cat and dog)
cifar10_train_data, cifar10_test_data = get_cifar10_2_class(trim, training_p)
ae_train_data = cifar10_train_data[:,:sub_train_set,:]
ae_test_data = cifar10_train_data[:,sub_train_set:,:]

# Define our pipeline
stage = 'dimensionReduction'
cv_factor = 5
algorithm1_pipe = Pipeline([(stage, algo1.Algorithm1(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
algorithm2_pipe = Pipeline([(stage, algo2.Algorithm2(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
lol_pipe = Pipeline([(stage, lol.Lol(r=r, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
passthrough_pipe = Pipeline([(stage, svcpass.Passthrough(r=latent_dim, verbose=False)),('scaler', StandardScaler()), ('svc', SVC())])
pipes = [algorithm1_pipe, algorithm2_pipe, lol_pipe]
last_xform = None

#class_1_data = cifar10_test_data[0]
#class_2_data = cifar10_test_data[1]

# Make and train the autoencoder
if use_autoencoder:
    pipes.append(passthrough_pipe)

    if type_autoencoder == 'basic':
        print('Using Basic Autoencoder')
        shape = (vector_size,)
        cifar_autoencoder = autoencoder.Autoencoder(latent_dim, shape)

        cifar10_train_data = np.concatenate(cifar10_train_data)
        cifar10_test_data = np.concatenate(cifar10_test_data)
        ae_train_data = np.concatenate(ae_train_data)
        ae_test_data = np.concatenate(ae_test_data)

        cifar_autoencoder.train_autoencoder(ae_train_data, ae_test_data)

        cifar_autoencoder.encoder.summary()
        cifar_autoencoder.decoder.summary()

        # Convert the training data to a single set of labeled data
        # Using class 1 and class 2 for our labels
        training_data = cifar10_train_data
        training_labels = training_p * [1] + training_p * [2]

        # Convert the images with the autoencoder to the latent space
        training_data = cifar_autoencoder.encoder(training_data).numpy()
        class_1_data = cifar_autoencoder.encoder(cifar10_test_data[:test_p]).numpy()
        class_2_data = cifar_autoencoder.encoder(cifar10_test_data[test_p:]).numpy()

    elif type_autoencoder == 'convolutional':
        print('Using Convolutional Autoencoder')
        shape = (image_width, image_width, 3)
        cifar10_train_data = reconstruct_image(cifar10_train_data, (2 * training_p, image_width, image_width, 3))
        cifar10_test_data = reconstruct_image(cifar10_test_data, (2 * test_p, image_width, image_width, 3))
        ae_train_data = reconstruct_image(ae_train_data, (2*sub_train_set, image_width, image_width, 3))
        ae_test_data = reconstruct_image(ae_test_data, (2*sub_train_set, image_width, image_width, 3))
        cifar_autoencoder =cae.ConvAutoencoder(shape)

        cifar_autoencoder.train_autoencoder(ae_train_data, ae_test_data)

        cifar_autoencoder.encoder.summary()
        cifar_autoencoder.decoder.summary()

        # Convert the training data to a single set of labeled data
        # Using class 1 and class 2 for our labels
        # training_data = np.concatenate(cifar10_train_data)
        training_data = cifar10_train_data
        training_labels = training_p * [1] + training_p * [2]

        # Convert the images with the autoencoder to the latent space
        training_data = cifar_autoencoder.encoder(training_data).numpy().reshape(2 * training_p, 8 * 8 * 8)
        class_1_data = cifar_autoencoder.encoder(cifar10_test_data[:test_p]).numpy().reshape(test_p, 8 * 8 * 8)
        class_2_data = cifar_autoencoder.encoder(cifar10_test_data[test_p:]).numpy().reshape(test_p, 8 * 8 * 8)
    elif type_autoencoder == 'variational':
        print('Using Variational Autoencoder')
        shape = (image_width, image_width, 3)
        cifar10_train_data = reconstruct_image(cifar10_train_data, (2 * training_p, image_width, image_width, 3))
        cifar10_test_data = reconstruct_image(cifar10_test_data, (2 * test_p, image_width, image_width, 3))
        cifar_autoencoder = vae.VarConvAutoencoder(latent_dim, shape)

        cifar_autoencoder.train_autoencoder(cifar10_train_data)

        cifar_autoencoder.encoder.summary()
        cifar_autoencoder.decoder.summary()

        # Convert the training data to a single set of labeled data
        # Using class 1 and class 2 for our labels
        # training_data = np.concatenate(cifar10_train_data)
        training_data = cifar10_train_data
        training_labels = training_p * [1] + training_p * [2]

        # Convert the images with the autoencoder to the latent space
        [z_mean, z_log_var, z_data] = cifar_autoencoder.encoder(training_data)
        training_data = z_data.numpy().reshape(2 * training_p, latent_dim)
        class_1_data = cifar_autoencoder.encoder(cifar10_test_data[:test_p])[2].numpy().reshape(test_p, latent_dim)
        class_2_data = cifar_autoencoder.encoder(cifar10_test_data[test_p:])[2].numpy().reshape(test_p, latent_dim)
else:
    training_data = np.concatenate(cifar10_train_data)
    training_labels = training_p * [1] + training_p * [2]
    class_1_data = cifar10_test_data[0]
    class_2_data = cifar10_test_data[1]

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
