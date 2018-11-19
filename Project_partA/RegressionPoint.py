import numpy as np


class RegressionPoint(object):
    """
    A class, which represent a single data point for the Linear Regression algorithm.
    self.Data = The features vector of each point
    self.Tag = The real tag of the point
    self.Class = The estimated tag according to the classifier
    self.classes_prob = The g values (result of the logistic regression function) vector of each point, for each class
    """

    def __init__(self, point: np.ndarray, tag: int = None, class_prob: np.ndarray = None, Class: int = 0):
        """
        Creates a RegressionPoint

        :param point: The features vector for the point.
        :param tag: Optional - The real tag of the point.
        :param class_prob: Optional - The results of the logistic function, for each class, for that point.
        :param Class: Optional - The class to which the classifier assigned this point, default value is 0.
        """

        self.Data = point
        self.classes_prob = class_prob
        self.Tag = tag
        self.Class = Class