import numpy as np


class ClusterPoint(object):
    """
    A class, which represent a single data point for the K-Means algorithm.
    self.data = The features vector of each point
    self.Cluster = The estimated cluster according to the classifier
    """

    def __init__(self, data, cluster=None):
        """
        Creates a ClusterPoint

        :param data: The features vector for the point.
        :param cluster: Optional - The current cluster to which the point belongs.
        """

        self.data = np.array(data).transpose()
        self.cluster = cluster
