import numpy as np
from ClusterPoint import ClusterPoint


class KMeans(object):
    """
    Our implementation of a class, which performs the K-Means algorithm, according to pre-determined hyper-parameters
    The user methods is: cluster
    """

    def __init__(self, k: int, tolerance=None, max_iterations=None):
        """
        Initializing the class
        :param k: Number of allowed clusters
        :param tolerance: Maximal error allowed
        :param max_iterations: Maximal number of iterations allowed
        """

        self._num_of_clusters = k
        self._clusters_centers = []
        self._data = []

        if tolerance is None:
            self._tolerance = 0.001

        else:
            self._tolerance = tolerance

        if max_iterations is None:
            self._max_iterations = 300

        else:
            self._max_iterations = max_iterations

    def cluster(self, data_set: np.ndarray, clusters_centers=None):
        """
        This is a user method. It clusters every example in a given data set, according to the K-Means algorithm
        and if available, uses the pre-determined starting locations for the clusters.

        :param data_set:  numpy array, size - M X N: The entire data set to be clustered. With M being the number of
                          features for each sample, and N being the number of samples.
        :param clusters_centers: Optional - The starting location of each cluster center
        :return: The locations of the all of the clusters' centers.
        """

        if clusters_centers is None:
            self._clusters_centers = self._init_clusters_centers(data_set, data_set.shape[1])

        else:
            self._clusters_centers = clusters_centers

        for i in range(data_set.shape[0]):
            p = self._cluster_point(ClusterPoint(data_set[i, :], 0), self._clusters_centers)
            self._data.extend([p])

        iteration = 1
        tolerance = 10 ** 9  # Initialize into a very large number
        current_centers = self._clusters_centers
        while iteration <= self._max_iterations:
            if tolerance <= self._tolerance:
                break

            j = 0
            for point in self._data:
                self._data[j] = self._cluster_point(point, self._clusters_centers)
                j += 1

            self._compute_centers(self._data)

            tolerance = np.mean(np.abs(current_centers - self._clusters_centers))

            iteration += 1

        data_clusters = np.zeros((data_set.shape[0], 1))
        j = 0
        for point in self._data:
            data_clusters[j] = point.cluster
            j += 1

        return data_clusters

    def _cluster_point(self, point: ClusterPoint, centers: np.ndarray):
        """
        An internal utility method.
        This method clusters a single ClusterPoint

        :param point: An initialized ClusterPoint object
        :param centers: The vector containing the centers of all clusters.
        :return: A newly clustered ClusterPoint, assigned to the closest center, in a euclidean sense, from the centers
                 vector
        """

        min_dist = 10 ** 9  # Very large number
        point.cluster = 0  # The default clustering
        for i in range(self._num_of_clusters):
            current_dist = self._compute_distance(point, centers[:, i])

            if current_dist < min_dist:
                min_dist = current_dist
                point.cluster = i

        return point

    def _compute_distance(self, point: ClusterPoint, center: np.ndarray):
        """
        An internal utility method.
        This method computes the distance of a ClusterPoint from a given cluster's center.

        :param point: An initialized ClusterPoint object.
        :param center: The centers of the cluster.
        :return: The distance, in a euclidean sense, of the point from the cluster's center.
        """

        dims = point.data.shape[0]
        dist = 0
        for i in range(dims):
            dist += (point.data[i] - center[i]) ** 2

        return dist

    def _compute_centers(self, data: list):
        """
        An internal utility method.
        This method computes the location of all of the clusters' centers, according to the points assigned to each
        center.

        :param data: A list, containing all of the ClusterPoints in the data set.
        :return: An M X N numpy array, containing all of the clusters' centers. With M being the dimension of each center
                 point, and N being the number of clusters.
        """

        for i in range(self._num_of_clusters):
            # Initialize location vector for the current center
            cluster_center = np.squeeze(np.zeros((data[0].data.shape[0], 1)))
            num_of_points = 0

            # Average the location of the cluster center, from the locations of all of the points belonging to it
            for point in data:
                if point.cluster == i:  # The point belongs to the current cluster
                    for dim in range(point.data.shape[0]):
                        cluster_center = np.add(cluster_center[:], point.data[:])

                    num_of_points += 1

                else:
                    continue

            if num_of_points != 0:
                self._clusters_centers[:, i] = cluster_center * (1/num_of_points)

            else:
                self._clusters_centers[:, i] = cluster_center

    def _init_clusters_centers(self, data_set: np.ndarray, dims: int):
        """
        An internal utility method.
        This method initializes the center points for all of the clusters, if no initialization is given by the user.

        :param data_set: numpy array, size - M X N: The entire data set to be clustered. With M being the number of
                         features for each sample, and N being the number of samples.
        :param dims: An int, representing the number of features for each sample.
        :return: An M X N numpy array, containing all of the clusters' centers. With M being the dimension of each center
                 point, and N being the number of clusters.
        """

        start = np.mean(data_set) - np.std(data_set)
        end = np.mean(data_set) + np.std(data_set)
        locs = np.linspace(start, end, self._num_of_clusters)
        clusters_centers = np.zeros((dims, self._num_of_clusters))
        for i in range(self._num_of_clusters):
            clusters_centers[:, i] = (np.ones((dims, 1)) * locs[i])[:, 0]

        return clusters_centers
