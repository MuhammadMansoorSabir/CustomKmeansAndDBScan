# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        X = check_array(X, accept_sparse='csr')
        np.random.seed(self.random_state)
        # suppose random cluster centers based on the number of k
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # calculate the distance of every point to each cluster center
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
            # after calculation assign each point to the nearest cluster center
            self.labels_ = np.argmin(distances, axis=0)
            # reposition cluster centers based on the mean position of each cluster
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            # if no significant change break the code
            if np.allclose(self.cluster_centers_, new_centers):
                break
            # update existing cluster centers with new centers
            self.cluster_centers_ = new_centers

        return self

    def fit_predict(self, X: np.ndarray, y=None):
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = X.astype(np.float32)
        # Get the number of samples in the dataset
        n_samples = X.shape[0]
        # Initialize labels_ with 0, marking all points as unclassified initially
        self.labels_ = np.full(n_samples, 0)
        cluster_label = 0

        for i in range(n_samples):
            # Check if the point has already been labeled, if yes, continue to the next point
            if self.labels_[i] != 0:
                continue
            # Find neighboring points within a specific radius (eps) around the current point
            neighbors = self._find_neighbors_within_radius(X, i)
            # Check if the length is less than the given minPts, then mark it as noise
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                # Assign a new cluster label and expand the cluster
                cluster_label += 1
                self.labels_[i] = cluster_label
                self._expand_cluster(X, i, neighbors, cluster_label)

        return self

    def _find_neighbors_within_radius(self, X, point_idx):
        # Calculate distances between the current point and all other points within the given radius(eps)
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_label):
        seeds = list(neighbors)

        while seeds:
            current_object = seeds.pop(0)
            # If the neighbor is not assigned to any cluster, assign it to the current cluster
            if self.labels_[current_object] == 0:
                self.labels_[current_object] = cluster_label
            # Find neighbors of the current neighbor
            current_neighbors = self._find_neighbors_within_radius(X, current_object)
            # If the number of neighbors is sufficient, add them to the cluster
            if len(current_neighbors) >= self.min_samples:
                for neighbor_index in current_neighbors:
                    if self.labels_[neighbor_index] in [-1, 0]:
                        seeds.append(neighbor_index)
                        self.labels_[neighbor_index] = cluster_label

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
