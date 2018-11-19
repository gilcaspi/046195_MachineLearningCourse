import scipy.io as sio
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from enum import Enum
import timeit


# ---------------------------------------- Define for each computer ------------------------------------
RAM_LIMIT = 3000


# ---------------------------------------- Enum Definitions --------------------------------------------
class ModelName(Enum):
    LINEAR = 0
    GAUSSIAN = 1
    POLY = 2


# ---------------------------------------- Function Definitions ----------------------------------------
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# ------------------------------------------ Pre - Processing ------------------------------------------
# Load the data
path = os.getcwd()
BreastCancerData = sio.loadmat(path + '\\' + 'BreastCancerData.mat')

# # Find Indexes for the benign vs. malignant tumors
# benign_inds = np.array(np.where(BreastCancerData['y'] == 0)).transpose()
# benign_inds = benign_inds[:, 0]
# malignant_inds = np.array(np.where(BreastCancerData['y'])).transpose()
# malignant_inds = malignant_inds[:, 0]
#
# # Separate the data to the two groups
# benign = BreastCancerData['X'][:, benign_inds]
# benign_tags = BreastCancerData['y'][benign_inds]
# malignant = BreastCancerData['X'][:, malignant_inds]
# malignant_tags = BreastCancerData['y'][malignant_inds]

# separate to test and train set
data = BreastCancerData['X']
tags = BreastCancerData['y']
train_inds2 = np.random.choice(np.arange(data.shape[1]), int(data.shape[1] * 0.8), replace=False)
test_inds2 = np.array(np.arange(data.shape[1]))
test_inds2 = np.delete(test_inds2, train_inds2)
data_for_train = data[:, train_inds2]
tags_for_train = tags[train_inds2]
data_for_test = data[:, test_inds2]
tags_for_test = tags[test_inds2]

# ------------------------------------------ SVM ------------------------------------------

# create the folds
# N_Splits = 10
N_Splits = 10
kf = KFold(n_splits=N_Splits, random_state=None, shuffle=False)

# create error lists for each model
linear_validation_error_list = []
gaussian_validation_error_list = []
poly_validation_error_list = []

linear_train_error_list = []
gaussian_train_error_list = []
poly_train_error_list = []
c_vec = [0.00000001, 0.000001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7,
         8, 25, 50, 100, 1000, 2500, 3000, 3750, 5000, 5600, 6250, 7000, 7500, 8750, 10000, 20000, 40000, 45000, 47500,
         50000, 52500, 55000, 60000, 70000, 80000, 90000, 100000, 200000,  1000000]
poly_degree = 3
# perform cross validation (on the train set) with K=10 folds for each hyper parameter C
for c in c_vec:
    # Print Hyper parameter
    print('Cross Validation with C = ' + str(c) + '\n')
    fold_num = 1
    for train_index, validation_index in kf.split(data_for_train.transpose()):
        # Print iteration
        print('Started Iteration over fold number ' + str(fold_num) + '\n')

        # split data
        train_data = data_for_train[:, train_index].transpose()
        train_tags = tags_for_train[train_index]
        validation_data = data_for_train[:, validation_index].transpose()
        validation_tags = tags_for_train[validation_index]

        # train
        # create models
        max_iter = 50000
        linear_kernel_svm = svm.SVC(C=c, kernel='linear', probability=False, max_iter=max_iter, cache_size=RAM_LIMIT)
        gaussian_kernel_svm = svm.SVC(gamma=1/c, kernel='rbf', max_iter=max_iter, cache_size=RAM_LIMIT)
        polynomial_kernel_svm = svm.SVC(coef0=0, kernel='poly', degree=np.round(c)+1, probability=False, max_iter=max_iter,
                                        cache_size=RAM_LIMIT)

        # train models
        linear_kernel_svm.fit(train_data, train_tags.ravel())
        gaussian_kernel_svm.fit(train_data, train_tags.ravel())
        polynomial_kernel_svm.fit(train_data, train_tags.ravel())

        # predict on training set
        linear_kernel_train_prediction = linear_kernel_svm.predict(train_data)
        gaussian_kernel_train_prediction = gaussian_kernel_svm.predict(train_data)
        polynomial_kernel_train_prediction = polynomial_kernel_svm.predict(train_data)

        # evaluate error on training set
        linear_train_error = 0
        gaussian_train_error = 0
        poly_train_error = 0

        n_train = train_tags.shape[0]
        for sample in range(n_train):
            if linear_kernel_train_prediction[sample] != train_tags[sample]:
                linear_train_error += 1
            if gaussian_kernel_train_prediction[sample] != train_tags[sample]:
                gaussian_train_error += 1
            if polynomial_kernel_train_prediction[sample] != train_tags[sample]:
                poly_train_error += 1

        # add errors on validation set to errors list for each model
        linear_train_error_list.append(linear_train_error/n_train)
        gaussian_train_error_list.append(gaussian_train_error/n_train)
        poly_train_error_list.append(poly_train_error/n_train)

        # predict on validation set
        linear_kernel_svm_prediction = linear_kernel_svm.predict(validation_data)
        gaussian_kernel_svm_prediction = gaussian_kernel_svm.predict(validation_data)
        polynomial_kernel_svm_prediction = polynomial_kernel_svm.predict(validation_data)

        # evaluate error on validation set
        linear_validation_error = 0
        gaussian_validation_error = 0
        poly_validation_error = 0

        n = validation_tags.shape[0]
        for sample in range(n):
            if linear_kernel_svm_prediction[sample] != validation_tags[sample]:
                linear_validation_error += 1
            if gaussian_kernel_svm_prediction[sample] != validation_tags[sample]:
                gaussian_validation_error += 1
            if polynomial_kernel_svm_prediction[sample] != validation_tags[sample]:
                poly_validation_error += 1

        # add errors on validation set to errors list for each model
        linear_validation_error_list.append(linear_validation_error/n)
        gaussian_validation_error_list.append(gaussian_validation_error/n)
        poly_validation_error_list.append(poly_validation_error/n)

        # update fold number
        fold_num += 1

print('Finished Validation for SVM')

# find best model and best C
linear_validation_error_mat = np.reshape(linear_validation_error_list, (len(c_vec), N_Splits))
linear_validation_error_for_c = np.mean(linear_validation_error_mat, axis=1)
linear_validation_best_c_index = np.argmin(linear_validation_error_for_c)
linear_validation_best_error = linear_validation_error_for_c[linear_validation_best_c_index]
linear_validation_best_c = c_vec[linear_validation_best_c_index]

gaussian_validation_error_mat = np.reshape(gaussian_validation_error_list, (len(c_vec), N_Splits))
gaussian_validation_error_for_c = np.mean(gaussian_validation_error_mat, axis=1)
gaussian_validation_best_c_index = np.argmin(gaussian_validation_error_for_c)
gaussian_validation_best_error = gaussian_validation_error_for_c[gaussian_validation_best_c_index]
gaussian_validation_best_c = c_vec[gaussian_validation_best_c_index]

poly_validation_error_mat = np.reshape(poly_validation_error_list, (len(c_vec), N_Splits))
poly_validation_error_for_c = np.mean(poly_validation_error_mat, axis=1)
poly_validation_best_c_index = np.argmin(poly_validation_error_for_c)
poly_validation_best_error = poly_validation_error_for_c[poly_validation_best_c_index]
poly_validation_best_c = c_vec[poly_validation_best_c_index]

best_model_vec = [linear_validation_best_error, gaussian_validation_best_error, poly_validation_best_error]
best_model_num = np.argmin(best_model_vec)

# Perform PCA
# pre-processing
# center data for train
mu_data_for_train = np.mean(data_for_train, axis=1)
mu_matrix = np.dot(mu_data_for_train.reshape((data_for_train.shape[0]), 1), np.ones((1, tags_for_train.shape[0])))
centered_data_for_train = data_for_train - mu_matrix
# normalize data for train
normalized_data_for_train = centered_data_for_train
# for i in range(centered_data_for_train.shape[0]):
#     normalized_data_for_train[i, :] = (normalized_data_for_train[i, :] / np.var(centered_data_for_train[i, :]))
left_eigen_vecs, singular_vals, right_eigen_vecs = np.linalg.svd(normalized_data_for_train)
PC1 = np.dot(left_eigen_vecs[:, 0], normalized_data_for_train)
PC2 = np.dot(left_eigen_vecs[:, 1], normalized_data_for_train)

# center data for test
mu_data_for_test = np.mean(data_for_test, axis=1)
mu_matrix_for_test = np.dot(mu_data_for_test.reshape((data_for_test.shape[0]), 1), np.ones((1, tags_for_test.shape[0])))
centered_data_for_test = data_for_test - mu_matrix_for_test
# normalize data for train
normalized_data_for_test = centered_data_for_test
# for i in range(centered_data_for_test.shape[0]):
#     normalized_data_for_test[i, :] = (normalized_data_for_test[i, :] / np.var(centered_data_for_test[i, :]))
left_eigen_vecs_test, singular_vals_test, right_eigen_vecs_test = np.linalg.svd(normalized_data_for_test)
PC1_for_test = np.dot(left_eigen_vecs_test[:, 0], normalized_data_for_test)
PC2_for_test = np.dot(left_eigen_vecs_test[:, 1], normalized_data_for_test)


# use np.concatinate to unite pc1 and pc2
PCs_mat = np.array([PC1, PC2]).transpose()
PCs_mat_for_test = np.array([PC1_for_test, PC2_for_test]).transpose()

# create model with best c for each kernel
max_iter = -1
best_linear_model_svm = svm.SVC(C=linear_validation_best_c, kernel='linear', probability=False, max_iter=max_iter,
                                cache_size=RAM_LIMIT)
best_linear_model_svm_for_test = svm.SVC(C=linear_validation_best_c, kernel='linear', probability=False,
                                         max_iter=max_iter,
                                         cache_size=RAM_LIMIT)
max_iter = 50000
best_gaussian_model_svm = svm.SVC(gamma=1/gaussian_validation_best_c, kernel='rbf', max_iter=max_iter,
                                  cache_size=RAM_LIMIT)
max_iter = 500000
best_gaussian_model_svm_for_test = svm.SVC(gamma=1/gaussian_validation_best_c, kernel='rbf', max_iter=max_iter,
                                           cache_size=RAM_LIMIT)
# max_iter = 500000
best_poly_model_svm = svm.SVC(coef0=0, kernel='poly', probability=False, degree=np.round(poly_validation_best_c)+1,
                              max_iter=max_iter, cache_size=RAM_LIMIT)
max_iter = 500000
best_poly_model_svm_for_test = svm.SVC(coef0=0, kernel='poly', degree=np.round(poly_validation_best_c)+1,
                                       probability=False, max_iter=max_iter, cache_size=RAM_LIMIT)

# evaluate best models

# train best model on all the train set
data_for_train = data_for_train.transpose()

start = timeit.default_timer()
best_linear_model_svm_for_test.fit(data_for_train, tags_for_train.ravel())
end = timeit.default_timer()
linear_train_time = end - start

start = timeit.default_timer()
best_gaussian_model_svm_for_test.fit(data_for_train, tags_for_train.ravel())
end = timeit.default_timer()
gaussian_train_time = end - start

start = timeit.default_timer()
best_poly_model_svm_for_test.fit(data_for_train, tags_for_train.ravel())
end = timeit.default_timer()
poly_train_time = end - start

# predict on train set
start = timeit.default_timer()
best_linear_kernel_train_prediction = best_linear_model_svm_for_test.predict(data_for_train)
end = timeit.default_timer()
linear_prediction_time = end - start

start = timeit.default_timer()
best_gaussian_kernel_train_prediction = best_gaussian_model_svm_for_test.predict(data_for_train)
end = timeit.default_timer()
gaussian_prediction_time = end - start

start = timeit.default_timer()
best_poly_kernel_train_prediction = best_poly_model_svm_for_test.predict(data_for_train)
end = timeit.default_timer()
poly_prediction_time = end - start

# evaluate best model on train set
best_linear_train_error = 0
best_gaussian_train_error = 0
best_poly_train_error = 0

n = tags_for_train.shape[0]
for sample in range(n):
    if best_linear_kernel_train_prediction[sample] != tags_for_train[sample]:
        best_linear_train_error += 1
    if best_gaussian_kernel_train_prediction[sample] != tags_for_train[sample]:
        best_gaussian_train_error += 1
    if best_poly_kernel_train_prediction[sample] != tags_for_train[sample]:
        best_poly_train_error += 1

best_linear_train_error = best_linear_train_error/n
best_gaussian_train_error = best_gaussian_train_error/n
best_poly_train_error = best_poly_train_error/n

# predict on test set
data_for_test = data_for_test.transpose()
best_linear_kernel_test_prediction = best_linear_model_svm_for_test.predict(data_for_test)
best_gaussian_kernel_test_prediction = best_gaussian_model_svm_for_test.predict(data_for_test)
best_poly_kernel_test_prediction = best_poly_model_svm_for_test.predict(data_for_test)

# evaluate best model on test set
best_linear_test_error = 0
best_gaussian_test_error = 0
best_poly_test_error = 0

n = tags_for_test.shape[0]
for sample in range(n):
    if best_linear_kernel_test_prediction[sample] != tags_for_test[sample]:
        best_linear_test_error += 1
    if best_gaussian_kernel_test_prediction[sample] != tags_for_test[sample]:
        best_gaussian_test_error += 1
    if best_poly_kernel_test_prediction[sample] != tags_for_test[sample]:
        best_poly_test_error += 1

best_linear_test_error = best_linear_test_error/n
best_gaussian_test_error = best_gaussian_test_error/n
best_poly_test_error = best_poly_test_error/n


# print error results
print('Print Error Results')
print('\n ---------------------------------------------------- \n')
print('Results for - SVM with Linear Kernel and C = '+str(linear_validation_best_c)+'\n')
print('Train Set Error = ' + str(best_linear_train_error) + '\n')
print('Train Time = ' + str(linear_train_time) + '\n')
print('Test Set Error = ' + str(best_linear_test_error) + '\n')
print('Prediction Time = ' + str(linear_prediction_time) + '\n')
print('\n ---------------------------------------------------- \n')
print('Results for - SVM with Gaussian Kernel and Gamma = '+str(1/gaussian_validation_best_c)+'\n')
print('Train Set Error = ' + str(best_gaussian_train_error) + '\n')
print('Train Time = ' + str(gaussian_train_time) + '\n')
print('Test Set Error = ' + str(best_gaussian_test_error) + '\n')
print('Prediction Time = ' + str(gaussian_prediction_time) + '\n')
print('\n ---------------------------------------------------- \n')
print('Results for - SVM with Polynomial Kernel Coef0 = '+str(0)+' and Degree = '
      + str(np.round(poly_validation_best_c)+1)+'\n')
print('Train Set Error = ' + str(best_poly_train_error) + '\n')
print('Train Time = ' + str(poly_train_time) + '\n')
print('Test Set Error = ' + str(best_poly_test_error) + '\n')
print('Prediction Time = ' + str(poly_prediction_time) + '\n')
print('\n ---------------------------------------------------- \n')

# Visualization
# create the best model
if best_model_num == ModelName.LINEAR.value:
    best_model_svm = best_linear_model_svm
elif best_model_num == ModelName.GAUSSIAN.value:
    best_model_svm = best_gaussian_model_svm
else:  # poly
    best_model_svm = best_poly_model_svm

print('Training the best models')
tags_for_train = np.squeeze(np.array(tags_for_train))
tags_for_test = np.squeeze(np.array(tags_for_test))

print('Training the linear model')
linear_model = best_linear_model_svm.fit(PCs_mat, tags_for_train)
support_vectors_linear_model = linear_model.support_vectors_
print('Training the gaussian model')
gaussian_model = best_gaussian_model_svm.fit(PCs_mat, tags_for_train)
support_vectors_gaussian_model = gaussian_model.support_vectors_
print('Training the polynomial model')
poly_model = best_poly_model_svm.fit(PCs_mat, tags_for_train)
support_vectors_poly_model = poly_model.support_vectors_

# title for the plots
titles = ('SVM with Linear Kernel and C = ' + str(linear_validation_best_c),
          'SVM with Gaussian Kernel and C = ' + str(gaussian_validation_best_c),
          'SVM with Polynomial Kernel and C = ' + str(poly_validation_best_c))

print('Plotting on Train set')
# X0, X1 = PC1_for_test, PC2_for_test
# y = tags_for_test
X0, X1 = PC1, PC2
y = tags_for_train
# Set-up 2x2 grid for plotting.
print('Creating the grid')
xx, yy = make_meshgrid(X0[:], X1[:], 1)


fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
plot_contours(ax1, linear_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
sv1 = ax1.scatter(support_vectors_linear_model[:, 0], support_vectors_linear_model[:, 1], c='k', edgecolors='k', marker='s')
t1 = ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_xticks(())
ax1.set_yticks(())
ax1.set_title(titles[0])
plt.legend([sv1, t1], ['Support Vectors', 'Real Tag - benign/malignant'])

plot_contours(ax2, gaussian_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax2.scatter(support_vectors_gaussian_model[:, 0], support_vectors_gaussian_model[:, 1], c='k', edgecolors='k', marker='s')
ax2.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_xticks(())
ax2.set_yticks(())
ax2.set_title(titles[1])


plot_contours(ax3, poly_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax3.scatter(support_vectors_poly_model[:, 0], support_vectors_poly_model[:, 1], c='k', edgecolors='k', marker='s')
ax3.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax3.set_xlim(xx.min(), xx.max())
ax3.set_ylim(yy.min(), yy.max())
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_xticks(())
ax3.set_yticks(())
ax3.set_title(titles[2])


plt.show()

print('Plotting on Test set')
X0, X1 = PC1_for_test, PC2_for_test
y = tags_for_test
# X0, X1 = PC1, PC2
# y = tags_for_train
# Set-up 2x2 grid for plotting.
print('Creating the grid')
xx, yy = make_meshgrid(X0[:], X1[:], 1)


fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
plot_contours(ax1, linear_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
sv1 = ax1.scatter(support_vectors_linear_model[:, 0], support_vectors_linear_model[:, 1], c='k', edgecolors='k', marker='s')
t1 = ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_xticks(())
ax1.set_yticks(())
ax1.set_title(titles[0])
plt.legend([sv1, t1], ['Support Vectors', 'Real Tag - benign/malignant'])

plot_contours(ax2, gaussian_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax2.scatter(support_vectors_gaussian_model[:, 0], support_vectors_gaussian_model[:, 1], c='k', edgecolors='k', marker='s')
ax2.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_xticks(())
ax2.set_yticks(())
ax2.set_title(titles[1])

plot_contours(ax3, poly_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax3.scatter(support_vectors_poly_model[:, 0], support_vectors_poly_model[:, 1], c='k', edgecolors='k', marker='s')
ax3.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax3.set_xlim(xx.min(), xx.max())
ax3.set_ylim(yy.min(), yy.max())
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_xticks(())
ax3.set_yticks(())
ax3.set_title(titles[2])

plt.show()

