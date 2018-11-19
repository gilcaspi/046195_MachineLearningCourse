import numpy as np
from RegressionPoint import RegressionPoint


class LinearClassifier(object):
    """
    Our implementation of a class, which performs the Linear Regression classification algorithm,
    according to pre-determined hyper-parameters
    The user methods are:
        classify - classifies a RegressionPoint object.
        train - Train the classifier over a given data set.
        evaluate - Evaluates the performance of the classifier over a list of RegressionPoints to be classified
    """

    def __init__(self, max_iterations: int, tolerance: float, training_method: str, learning_rate: float,
                 num_of_classes: int = 2):
        """
        Initializing the classifier.

        :param max_iterations: Int - The maximal number of iterations allowed.
        :param tolerance: Float - The maximal error allowed.
        :param training_method: Str - The method by which to perform the gradient descent. Either 'Serial' for a serial
                                gradient descent, or 'Batch' for the 'Batch' version of gradient descent.
        :param learning_rate: Float - The learning rate parameter for the gradient descent.
        :param num_of_classes: Optional - Int - The number of possible classes. The default value is 2.
        """

        # Setup
        self._weights = np.zeros([])
        self.classes = np.arange(start=0, stop=num_of_classes, step=1)
        self._data_points = []
        self._method = training_method
        self._num_of_samples = 0
        self._max_iterations = max_iterations
        self._tolerance = float(tolerance)
        self._learning_rate = float(learning_rate)
        self.convergence_err = []

    def classify(self, point: RegressionPoint):
        """
        A user method.
        This method classifies a RegressionPoint object into one of the possible classes.

        :param point: RegressionPoint - The point to be classified.
        :return: RegressionPoint - The point to be classified.
        """

        # Calculate all of the g_k functions
        probs = []
        for k in range(self.classes.shape[0] - 1):
            probs.extend([self._calc_logistic_func(point, k)])

        probs.extend([1 - np.sum(probs)])
        probs = np.array(probs)

        # Update point
        point.classes_prob = probs
        Class = np.argmax(probs)
        point.Class = Class

        return point

    def train(self, training_data: np.ndarray, tags: np.ndarray):
        """
        A user method.
        This method trains the classifier over a given training set.

        :param training_data: Numpy array, size M X N: with M being the number of features for each point, and N being
                              the number of samples. - Contains the entire training set
        :param tags: Numpy array, size N X 1: With N being the number of samples. - Contains the true tags for each
                     sample.
        :return: None
        """

        self._num_of_samples = training_data.shape[1]
        self._weights = np.zeros([training_data.shape[0] + 1, self.classes.shape[0] - 1])

        # Normalize the training data
        for i in range(training_data.shape[0]):
            training_data[i, :] = training_data[i, :] - np.mean(training_data[i, :])
            training_data[i, :] = training_data[i, :] * 1/(np.max(training_data[i, :]))

        for i in range(self._num_of_samples):
            data = training_data[:, i]
            data = np.insert(data, 0, 1)  # Insert 1 at the start, for the bias element

            point = RegressionPoint(data, 1)
            point.Tag = int(tags[i])
            self.classify(point)
            self._data_points.extend([point])

        if self._method == 'Batch':
            self._batch_learning()

        elif self._method == 'Serial':
            self._serial_learning()

    def _calc_logistic_func(self, point: RegressionPoint, k):
        """
        An inner utility method.
        This method applies the logistic function for a specific class, over a data point,

        :param point: RegressionPoint - The point on which to perform the calculations.
        :param k: Int - The class for which to perform the calculation.
        :return: Float - The result of the logistic function.
        """

        nominator = np.exp(np.dot(self._weights[:, k].transpose(), point.Data))

        denominator = 0
        for i in range(self.classes.shape[0] - 1):
            denominator = denominator + np.exp(np.dot(self._weights[:, i].transpose(), point.Data))

        # Add one for the final class K = 0
        denominator += 1

        g = nominator/denominator

        return g

    def _serial_learning(self):
        """
        An inner utility method.
        This method perform the training of the classifier using a serial gradient descent.

        :return: None
        """

        # Prepare for training
        tolerance = float(10 ** 6)  # Initialize to a large value
        iteration = 0
        inds = []
        best_weights = []
        best_tolerance = tolerance

        # Train
        # While end condition is not yet met
        while tolerance > self._tolerance:
            # Check another end condition
            if iteration > self._max_iterations:
                break

            # Choose a random point
            if inds.__len__() >= self._data_points.__len__():
                inds = []

            point_ind = int(np.random.random_integers(0, high=(self._data_points.__len__() - 1), size=1))

            # Re-sample data point if already chosen
            while point_ind in inds:
                point_ind = int(np.random.random_integers(0, high=(self._data_points.__len__() - 1), size=1))

            inds.extend([point_ind])
            point = self._data_points[point_ind]

            # Update Weights
            self._serial_update(point)

            # Re-classify the point
            point = self.classify(point)
            self._data_points[point_ind] = point

            # Calculate error
            tolerance = self._compute_error()
            self.convergence_err.extend([tolerance])

            # Save the best configuration
            if tolerance < best_tolerance:
                best_weights = self._weights

            # Update Counter
            iteration += 1

        self._weights = best_weights

    def _serial_update(self, point: RegressionPoint):
        """
        An inner utility method.
        This method updates the classifier's weights according to the serial gradient descent.

        :param point: RegressionPoint - The point by which to perform the serial gradient descent.
        :return: None
        """

        y_til = np.zeros(self.classes.shape[0], )
        y_til[point.Tag] = 1

        for k in range(self._weights.shape[1]):
            self._weights[:, k] = np.array(self._weights[:, k] + self._learning_rate *
                                           (y_til[k] - point.classes_prob[k]) * point.Data).astype(float)

    def _batch_learning(self):
        """
        An inner utility method.
        This method perform the training of the classifier using a batch gradient descent.

        :return: None
        """

        # Prepare for training
        tolerance = float(10 ** 6)  # Initialize to a large value
        iteration = 0
        best_weights = []
        best_tolerance = tolerance

        # Train
        # While end condition is not yet met
        while tolerance > self._tolerance:
            # Check end condition
            if iteration > self._max_iterations:
                break

            # Update Weights
            self._batch_update()

            # Re-classify the points
            for i in range(self._data_points.__len__()):
                point = self._data_points[i]
                point = self.classify(point)
                self._data_points[i] = point

            # Calculate error
            tolerance = self._compute_error()

            # Save the best configuration
            if tolerance < best_tolerance:
                best_weights = self._weights
                best_tolerance = tolerance

            self.convergence_err.extend([tolerance])

            iteration += 1

        self._weights = best_weights

    def _batch_update(self):
        """
        An inner utility method.
        This method updates the classifier's weights according to the batch gradient descent.

        :return: None
        """

        tmp_data = np.zeros([self._weights.shape[0], self._weights.shape[1]])
        N = self._data_points.__len__()

        for k in range(self._weights.shape[1]):
            for i in range(N):
                y_til = np.zeros(self.classes.shape[0], )
                y_til[self._data_points[i].Tag] = 1

                tmp_data[:, k] += (y_til[k] - self._data_points[i].classes_prob[k]) * self._data_points[i].Data

            self._weights[:, k] = np.array((1/N) * self._weights[:, k] +
                                           self._learning_rate * tmp_data[:, k]).astype(float)

        # Weights regularization
        if np.max(np.abs(self._weights)) > 1:
            self._weights = self._weights * (1/np.max(np.abs(self._weights)))

    def _compute_error(self):
        """
        An inner utility method.
        This method computes the error for the classifier over the entire training set.

        :return: Float - The error of the classifier.
        """

        # Calculate the total error
        error = 0
        for point in self._data_points:
            if point.Tag != point.Class:
                error += 1

        return error/self._data_points.__len__()

    def evaluate(self, points: list):
        """
        A user method.
        This method evaluates the error of the classifier over a list of RegressionPoints, which serves as a testing
        set.

        :param points: A list - Contains all of the RegressionPoints of the testing set.
        :return: Float - The classifier's error.
        """

        # Evaluate the error on a test set
        error = 0
        for point in points:
            if point.Tag != point.Class:
                error += 1

        return error/points.__len__()

