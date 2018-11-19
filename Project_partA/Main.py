import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from LogisticRegression import LinearClassifier
from RegressionPoint import RegressionPoint
import timeit
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
from pprint import pprint


# ------------------------------------------ Question 1 ------------------------------------------
print('\n ---------------------------------------------------- \n')
print('K-Means: \n')
# Load the data
path = os.getcwd()
BreastCancerData = sio.loadmat(path + '\\' + 'BreastCancerData.mat')

# Perform PCA
left_eigen_vecs, singular_vals, right_eigen_vecs = np.linalg.svd(BreastCancerData['X'])
PC1 = np.dot(left_eigen_vecs[:, 0], BreastCancerData['X'])
PC2 = np.dot(left_eigen_vecs[:, 1], BreastCancerData['X'])

# Find Indexes for the benign vs. malignant tumors
benign_inds = np.array(np.where(BreastCancerData['y'] == 0)).transpose()
benign_inds = benign_inds[:, 0]
malignant_inds = np.array(np.where(BreastCancerData['y'])).transpose()
malignant_inds = malignant_inds[:, 0]

# Plot Scatter With separation Plot
plt.figure()
b = plt.scatter(PC1[benign_inds], PC2[benign_inds], c='b')
m = plt.scatter(PC1[malignant_inds], PC2[malignant_inds], c='r')
plt.title('PCA Scatter Plot')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend([b, m], ['Benign - 0', 'Malignant - 1'])
plt.show()

# Separate the data to the two groups
benign = BreastCancerData['X'][:, benign_inds]
benign_tags = BreastCancerData['y'][benign_inds]
malignant = BreastCancerData['X'][:, malignant_inds]
malignant_tags = BreastCancerData['y'][malignant_inds]

# Randomly choose samples
data = BreastCancerData['X']
tags = BreastCancerData['y']
train_inds2 = np.random.choice(np.arange(data.shape[1]), int(data.shape[1] * 0.8), replace=False)
test_inds2 = np.array(np.arange(data.shape[1]))
test_inds2 = np.delete(test_inds2, train_inds2)
data_for_train = data[:, train_inds2]
tags_for_train = tags[train_inds2]
data_for_test = data[:, test_inds2]
tags_for_test = tags[test_inds2]

# Perform KMeans Clustering, K = 2
K = 2
km = KMeans(k=K, tolerance=0.01, max_iterations=300)

# The transpose is in order to fit to the required structure of KMeans
cluster_time_start = timeit.default_timer()
clustered_data = km.cluster(BreastCancerData['X'].transpose())
cluster_time_stop = timeit.default_timer()

print('Time for clustering via K-Means: ' + str(cluster_time_stop - cluster_time_start) + '\n')

# Calculate the error
error = 0
for i in range(clustered_data.shape[0]):
    if clustered_data[i] != tags[i]:
        error += 1

error /= int(clustered_data.shape[0])

print('K-Means algorithm error is: ' + str(error) + '\n')
print('\n ---------------------------------------------------- \n')

# Find the correct K cluster for each point
benign_k_cluster = clustered_data[benign_inds]
malignant_k_cluster = clustered_data[malignant_inds]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
bb = ax.scatter(PC1[benign_inds], PC2[benign_inds], benign_k_cluster, marker='o', c='b')
mm = ax.scatter(PC1[malignant_inds], PC2[malignant_inds], malignant_k_cluster, marker='x', c='r')
plt.title('K Clusters Plot Vs 2D PCA, K = 2')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('K-Cluster')
plt.legend([bb, mm], ['Benign - 0', 'Malignant - 1'])
plt.show()

# Perform KMeans Clustering, K = 4
K = 4
km = KMeans(k=K, tolerance=0.01, max_iterations=300)

# The transpose is in order to fit to the required structure of KMeans
clustered_data = km.cluster(BreastCancerData['X'].transpose())

# Find the indices of each cluster
cluster_inds = []
cluster_inds.extend([np.where(clustered_data == 0)[0]])  # Returns as an empty Class
cluster_inds.extend([np.where(clustered_data == 1)[0]])
cluster_inds.extend([np.where(clustered_data == 2)[0]])
cluster_inds.extend([np.where(clustered_data == 3)[0]])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c1 = ax.scatter(PC1[cluster_inds[1]], PC2[cluster_inds[1]], clustered_data[cluster_inds[1]], marker='o', c='b')
c2 = ax.scatter(PC1[cluster_inds[2]], PC2[cluster_inds[2]], clustered_data[cluster_inds[2]], marker='x', c='r')
c3 = ax.scatter(PC1[cluster_inds[3]], PC2[cluster_inds[3]], clustered_data[cluster_inds[3]], marker='s', c='g')
plt.title('K Clusters Plot Vs 2D PCA, K = 4')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('K-Cluster')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.show()

# Plot with Benign / Malignant annotations
c1_b = np.in1d(cluster_inds[1], benign_inds)
c1_m = np.in1d(cluster_inds[1], malignant_inds)
c2_b = np.in1d(cluster_inds[2], benign_inds)
c2_m = np.in1d(cluster_inds[2], malignant_inds)
c3_m = np.in1d(cluster_inds[3], malignant_inds)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c1 = ax.scatter(PC1[cluster_inds[1]][c1_b], PC2[cluster_inds[1]][c1_b], clustered_data[cluster_inds[1]][c1_b],
                marker='o', c='b')
c2 = ax.scatter(PC1[cluster_inds[2]][c2_b], PC2[cluster_inds[2]][c2_b], clustered_data[cluster_inds[2]][c2_b],
                marker='x', c='b')
c3 = ax.scatter(PC1[cluster_inds[1]][c1_m], PC2[cluster_inds[1]][c1_m], clustered_data[cluster_inds[1]][c1_m],
                marker='v', c='r')
c4 = ax.scatter(PC1[cluster_inds[2]][c2_m], PC2[cluster_inds[2]][c2_m], clustered_data[cluster_inds[2]][c2_m],
                marker='s', c='r')
c5 = ax.scatter(PC1[cluster_inds[3]][c3_m], PC2[cluster_inds[3]][c3_m], clustered_data[cluster_inds[3]][c3_m],
                marker='*', c='r')
plt.title('K Clusters Plot Vs 2D PCA, K = 4')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('K-Cluster')
plt.legend([c1, c2, c3, c4, c5], ['Cluster 1 - Benign', 'Cluster 2 - Benign', 'Cluster 1 - Malignant',
                                  'Cluster 2 - Malignant', 'Cluster 3 - Malignant'])
plt.show()

# ------------------------------------------ Question 2 ------------------------------------------

tags = BreastCancerData['y']
nb = NaiveBayes()

# train
nb_learn_start = timeit.default_timer()
nb.train(data_for_train, tags_for_train)
nb_learn_end = timeit.default_timer()

# evaluate
nb_train_error = nb.evaluate(data_for_train, tags_for_train)
nb_clf_start = timeit.default_timer()
nb_test_error = nb.evaluate(data_for_test, tags_for_test)
nb_clf_end = timeit.default_timer()

# Print the results
print('\n ---------------------------------------------------- \n')
print('Results for - Naive Bayes \n')
print('Train Set Error = ' + str(nb_train_error) + '.\n')
print('Test Set Error = ' + str(nb_test_error) + '.\n')
print('Train Time = ' + str(nb_learn_end-nb_learn_start) + '.\n')
print('Classification Time = ' + str(nb_clf_end-nb_clf_start) + '.\n')
print('\n ---------------------------------------------------- \n')

# ------------------------------------------ Question 3 ------------------------------------------
print('Logistic Regression: \n')
# Generate a linear, logistic regression, classifiers, with K = 2, with serial training, and different learning rates
Max_Iters = 200
serial_lc_1 = LinearClassifier(max_iterations=Max_Iters, tolerance=0.05, training_method='Serial', learning_rate=0.5,
                               num_of_classes=2)

serial_lc_2 = LinearClassifier(max_iterations=Max_Iters, tolerance=0.05, training_method='Serial', learning_rate=0.1,
                               num_of_classes=2)

serial_lc_3 = LinearClassifier(max_iterations=Max_Iters, tolerance=0.05, training_method='Serial', learning_rate=0.01,
                               num_of_classes=2)

# Generate a linear, logistic regression, classifiers, with K = 2, with batch training, and different learning rates
batch_lc_1 = LinearClassifier(max_iterations=Max_Iters, tolerance=0.05, training_method='Batch', learning_rate=0.001,
                              num_of_classes=2)

# Create the folds
N_Splits = 10
kf = KFold(n_splits=N_Splits, random_state=None, shuffle=False)

# Train and test the classifier
tags = tags_for_train
data = data_for_train

s_lc_1_errors = []
s_lc_2_errors = []
s_lc_3_errors = []
b_lc_1_errors = []

serial_timing = []
batch_timing = []

converg_errors_1_train = []
converg_errors_2_train = []
converg_errors_3_train = []
converg_errors_4_train = []

converg_errors_1_test = []
converg_errors_2_test = []
converg_errors_3_test = []

loop_iter = 1
for train_index, test_index in kf.split(data.transpose()):
    # Print iteration
    print('Started Iteration over fold number ' + str(loop_iter) + '\n')

    # Train the different classifiers
    training_data = data[:, train_index]
    training_tags = tags[train_index]

    # Prepare the testing set
    testing_data = data[:, test_index]
    testing_tags = tags[test_index]

    # Normalize the testing data
    for j in range(testing_data.shape[0]):
        testing_data[j, :] = testing_data[j, :] - np.mean(testing_data[j, :])
        testing_data[j, :] = testing_data[j, :] * (1 / np.max(testing_data[j, :]))

    start1 = timeit.default_timer()
    serial_lc_1.train(training_data, training_tags)
    stop1 = timeit.default_timer()
    serial_timing.extend([stop1 - start1])

    serial_lc_2.train(training_data, training_tags)
    serial_lc_3.train(training_data, training_tags)

    start2 = timeit.default_timer()
    batch_lc_1.train(training_data, training_tags)
    stop2 = timeit.default_timer()
    batch_timing.extend([stop2 - start2])

    # Get the error at each step
    converg_errors_1_train.extend([serial_lc_1.convergence_err])
    converg_errors_2_train.extend([serial_lc_2.convergence_err])
    converg_errors_3_train.extend([serial_lc_3.convergence_err])
    converg_errors_4_train.extend([batch_lc_1.convergence_err])

    points_s_1 = []
    points_s_2 = []
    points_s_3 = []
    points_b_1 = []

    for ind in (test_index - test_index[0]):  # Shift the indices between 0 and the num of test samples
        tmp_data = testing_data[:, ind]
        tmp_tag = testing_tags[ind]
        tmp_data = np.insert(tmp_data, 0, 1)  # Insert 1 at the start, for the bias element

        # Need to create 4 identical points, in order to prevent all classifiers to address the same object in memory
        # (like the need to use the "Copy" method in C++)
        point1 = RegressionPoint(tmp_data, tmp_tag)
        point2 = RegressionPoint(tmp_data, tmp_tag)
        point3 = RegressionPoint(tmp_data, tmp_tag)
        point4 = RegressionPoint(tmp_data, tmp_tag)

        clustering_time_start = timeit.default_timer()
        point1 = serial_lc_1.classify(point1)
        cluster_time_end = timeit.default_timer()

        if loop_iter == 1:
            print('Clustering Time: ' + str(cluster_time_end - clustering_time_start) + '\n')

        points_s_1.extend([point1])
        point2 = serial_lc_2.classify(point2)
        points_s_2.extend([point2])
        point3 = serial_lc_3.classify(point3)
        points_s_3.extend([point3])
        point4 = batch_lc_1.classify(point4)
        points_b_1.extend([point4])

        converg_errors_1_test.extend([serial_lc_1.evaluate([point1])])
        converg_errors_2_test.extend([serial_lc_2.evaluate([point2])])
        converg_errors_3_test.extend([serial_lc_3.evaluate([point3])])

    # Evaluate the different classifiers
    s_lc_1_errors.extend([serial_lc_1.evaluate(points_s_1)])
    s_lc_2_errors.extend([serial_lc_2.evaluate(points_s_2)])
    s_lc_3_errors.extend([serial_lc_3.evaluate(points_s_3)])
    b_lc_1_errors.extend([batch_lc_1.evaluate(points_b_1)])

    # Update the loop iterator
    loop_iter += 1

# Calculate the classifiers statistics
s_lc_1_errors = np.array(s_lc_1_errors)
s_lc_2_errors = np.array(s_lc_2_errors)
s_lc_3_errors = np.array(s_lc_3_errors)
b_lc_1_errors = np.array(b_lc_1_errors)

serial_err_1_mean = np.mean(s_lc_1_errors)
serial_err_1_std = np.std(s_lc_1_errors)
serial_err_2_mean = np.mean(s_lc_2_errors)
serial_err_2_std = np.std(s_lc_2_errors)
serial_err_3_mean = np.mean(s_lc_3_errors)
serial_err_3_std = np.std(s_lc_3_errors)
batch_err_1_mean = np.mean(b_lc_1_errors)
batch_err_1_std = np.std(b_lc_1_errors)

# Print running times
print('\n ---------------------------------------------------- \n')
print('The mean training time for the "serial" version of the algorithm is: ' + str(np.mean(serial_timing)) + '\n')
print('The mean training time for the "batch" version of the algorithm is: ' + str(np.mean(batch_timing)) + '\n')

# Print the mean results and std
print('\n ---------------------------------------------------- \n')
print('Results for - Linear Regression, Serial training method, learning rate = 0.5: \n')
print('Mean Error = ' + str(serial_err_1_mean) + ' , std = ' + str(serial_err_1_std) + '.\n')

print('\n ---------------------------------------------------- \n')
print('Results for - Linear Regression, Serial training method, learning rate = 0.1: \n')
print('Mean Error = ' + str(serial_err_2_mean) + ' , std = ' + str(serial_err_2_std) + '.\n')

print('\n ---------------------------------------------------- \n')
print('Results for - Linear Regression, Serial training method, learning rate = 0.01: \n')
print('Mean Error = ' + str(serial_err_3_mean) + ' , std = ' + str(serial_err_3_std) + '.\n')

print('\n ---------------------------------------------------- \n')
print('Results for - Linear Regression, Batch training method, learning rate = 0.001: \n')
print('Mean Error = ' + str(batch_err_1_mean) + ' , std = ' + str(batch_err_1_std) + '.\n')

# Plot Total Results of errors over the training set in each fold
iterations = np.arange(N_Splits) + 1
ax = plt.subplot(111)
p1 = ax.plot(iterations, s_lc_1_errors, c='b')
p2 = ax.plot(iterations, s_lc_2_errors, c='r')
p3 = ax.plot(iterations, s_lc_3_errors, c='g')
p4 = ax.plot(iterations, b_lc_1_errors, c='c')
ax.set_xlabel('Fold Number')
ax.set_ylabel('Error')
plt.show()

# Plot Convergence Results over the training set in each fold, for the serial classifiers
train_iters = np.arange(Max_Iters) + 1

plt.figure()
plt.plot(train_iters, np.array(converg_errors_1_train[0])[0:Max_Iters], c='b')
plt.plot(train_iters, np.array(converg_errors_2_train[0])[0:Max_Iters], c='r')
plt.plot(train_iters, np.array(converg_errors_3_train[0])[0:Max_Iters], c='g')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.show()

# Plot results for the validation set, for the serial classifiers
plt.figure()
plt.plot(iterations, s_lc_1_errors, c='b')
plt.plot(iterations, s_lc_2_errors, c='r')
plt.plot(iterations, s_lc_3_errors, c='g')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.show()

# Plot the convergence error for the validation set, for the Batch classifiers
plt.figure()
plt.plot(train_iters, np.array(converg_errors_4_train[0])[0:Max_Iters], c='b')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.show()

# Plot results for the validation set, for the Batch classifiers
plt.figure()
plt.plot(iterations, b_lc_1_errors, c='r')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.show()

# Evaluate the results for the test set
# Choose the best model
models = [serial_lc_1, serial_lc_2, serial_lc_3]
errors = np.array([serial_err_1_mean, serial_err_2_mean, serial_err_3_mean])
best_model = np.argmin(errors)
model = models[best_model]

# Train the best model over the entire training set
Max_itarations = 500
model._max_iterations = Max_itarations
model.train(data_for_train, tags_for_train)

# Prepare the testing set
# Normalize the testing data
norm_data_for_test = data_for_test
for j in range(data_for_test.shape[0]):
    norm_data_for_test[j, :] = data_for_test[j, :] - np.mean(data_for_test[j, :])
    norm_data_for_test[j, :] = data_for_test[j, :] * (1 / np.max(data_for_test[j, :]))

test_regression_points = []
for p in range(tags_for_test.__len__()):
    tmp_data = norm_data_for_test[:, p]
    tmp_tag = tags_for_test[p]
    tmp_data = np.insert(tmp_data, 0, 1)
    tmp_point = RegressionPoint(tmp_data, tmp_tag)
    clustering_time_start = timeit.default_timer()
    tmp_point = model.classify(tmp_point)
    cluster_time_end = timeit.default_timer()
    test_regression_points.extend([tmp_point])

# Evaluate
regression_model_error = model.evaluate(test_regression_points)

# Print the mean results and std
print('\n ---------------------------------------------------- \n')
print('Regression Classifier results, over the testing set: \n')
print('Error = ' + str(regression_model_error) + '.\n')
print('\n ---------------------------------------------------- \n')
print('\n')

# ------------------------------------------ Question 4 ------------------------------------------

kf = KFold(n_splits=10, random_state=None, shuffle=False)
data = data_for_train
tags = tags_for_train
dt_train_error_entropy = []
dt_test_error_entropy = []
dt_train_error_gini = []
dt_test_error_gini = []
dt_train_error_miss = []
dt_test_error_miss = []
dt_learn_time_ent = []
dt_clf_time_ent = []
dt_learn_time_gini = []
dt_clf_time_gini = []
dt_learn_time_miss = []
dt_clf_time_miss = []
fold_num = 1
for train_index, test_index in kf.split(data.transpose()):
    # Print iteration
    print('Started Iteration over fold number ' + str(fold_num) + '\n')

    # split data
    train_data = data[:, train_index].transpose()
    test_data = data[:, test_index].transpose()

    # train
    start_learn_ent = timeit.default_timer()
    entropy_tree = DecisionTree("entropy", 0, 10)
    entropy_tree.train(train_data, tags[train_index])
    end_learn_ent = timeit.default_timer()
    dt_learn_time_ent.append(end_learn_ent - start_learn_ent)
    # evaluate
    start_clf_ent = timeit.default_timer()
    ent_decision_tree_train_error = entropy_tree.evaluate(train_data, tags[train_index])
    end_clf_ent = timeit.default_timer()
    dt_train_error_entropy.append(ent_decision_tree_train_error)
    ent_decision_tree_test_error = entropy_tree.evaluate(test_data, tags[test_index])
    dt_test_error_entropy.append(ent_decision_tree_test_error)
    dt_clf_time_ent.append(end_clf_ent - start_clf_ent)

    # train
    start_learn_gini = timeit.default_timer()
    gini_tree = DecisionTree('gini', 0, 10)
    gini_tree.train(train_data, tags[train_index])
    end_learn_gini = timeit.default_timer()
    dt_learn_time_gini.append(end_learn_gini - start_learn_gini)
    # evaluate
    start_clf_gini = timeit.default_timer()
    dt_train_error_gini.append(gini_tree.evaluate(train_data, tags[train_index]))
    dt_test_error_gini.append(gini_tree.evaluate(test_data, tags[test_index]))
    end_clf_gini = timeit.default_timer()
    dt_clf_time_gini.append(end_clf_gini - start_clf_gini)
    # train
    start_learn_miss = timeit.default_timer()
    miss_tree = DecisionTree('misclassification', 0.01, 10)
    miss_tree.train(train_data, tags[train_index])
    end_learn_miss = timeit.default_timer()
    dt_learn_time_miss.append(end_learn_miss - start_learn_miss)
    # evaluate
    start_clf_miss = timeit.default_timer()
    dt_train_error_miss.append(miss_tree.evaluate(train_data, tags[train_index]))
    end_clf_miss = timeit.default_timer()
    dt_clf_time_miss.append(end_clf_miss - start_clf_miss)
    dt_test_error_miss.append(miss_tree.evaluate(test_data, tags[test_index]))

    # Update the loop iterator
    fold_num += 1
print('Finished Validation for Decision Tree ')

# Test
model_error = [(np.mean(dt_test_error_entropy)), np.mean(dt_test_error_gini), np.mean(dt_test_error_miss)]
best_tree_ind = np.argmin(model_error)
if best_tree_ind == 0:
    best_tree = DecisionTree("entropy", 0, 10)
    best_model_name = "Entropy Criterion"
elif best_tree_ind == 1:
    best_tree = DecisionTree('gini', 0, 10)
    best_model_name = "Gini Criterion"
else:
    best_tree = DecisionTree('misclassification', 0.01, 10)
    best_model_name = "Misclassification Criterion"
# train on all train data
best_tree.train(data_for_train.transpose(), tags_for_train)
best_tree_depth = best_tree.get_depth()
dt_best_model_test_error = best_tree.evaluate(data_for_test.transpose(), tags_for_test)

# Print the results
print('\n ---------------------------------------------------- \n')
print('Results for - Decision Tree with Entropy Criterion\n')
print('Mean Train Set Error = ' + str(np.mean(dt_train_error_entropy))+' , std = ' + str(np.std(dt_train_error_entropy))
      + '.\n')
print('Mean Test Set Error = ' + str(np.mean(dt_test_error_entropy))+' , std = ' + str(np.std(dt_test_error_entropy))
      + '.\n')
print('Mean Train Time = ' + str(np.mean(dt_learn_time_ent))+' , std = ' + str(np.std(dt_learn_time_ent))
      + '.\n')
print('Mean Classification Time = ' + str(np.mean(dt_clf_time_ent))+' , std = ' + str(np.std(dt_clf_time_ent))
      + '.\n')
print('\n ---------------------------------------------------- \n')

print('\n ---------------------------------------------------- \n')
print('Results for - Decision Tree with Gini Criterion\n')
print('Mean Train Set Error = ' + str(np.mean(dt_train_error_gini))+' , std = ' + str(np.std(dt_train_error_gini))
      + '.\n')
print('Mean Test Set Error = ' + str(np.mean(dt_test_error_gini))+' , std = ' + str(np.std(dt_test_error_gini))
      + '.\n')
print('Mean Train Time = ' + str(np.mean(dt_learn_time_gini))+' , std = ' + str(np.std(dt_learn_time_gini))
      + '.\n')
print('Mean Classification Time = ' + str(np.mean(dt_clf_time_gini))+' , std = ' + str(np.std(dt_clf_time_gini))
      + '.\n')
print('\n ---------------------------------------------------- \n')

print('\n ---------------------------------------------------- \n')
print('Results for - Decision Tree with Misclassification Criterion\n')
print('Mean Train Set Error = ' + str(np.mean(dt_train_error_miss))+' , std = ' + str(np.std(dt_train_error_miss))
      + '.\n')
print('Mean Test Set Error = ' + str(np.mean(dt_test_error_miss))+' , std = ' + str(np.std(dt_test_error_miss))
      + '.\n')
print('Mean Train Time = ' + str(np.mean(dt_learn_time_miss))+' , std = ' + str(np.std(dt_learn_time_miss))
      + '.\n')
print('Mean Classification Time = ' + str(np.mean(dt_clf_time_miss))+' , std = ' + str(np.std(dt_clf_time_miss))
      + '.\n')
print('\n ---------------------------------------------------- \n')

print('\n ---------------------------------------------------- \n')
print('Results for - Decision Tree Best Model\n')
print('Best Model = ' + best_model_name)
print('Test Set Error = ' + str(np.mean(dt_train_error_miss)) + '.\n')
print('Test Depth = ' + str(best_tree_depth) + '.\n')
print('\n ---------------------------------------------------- \n')
