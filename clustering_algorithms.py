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

        # Initialize centroids randomly
        np.random.seed(self.random_state)
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Calculate distances from each point to centroids
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
            # distances = np.sqrt(np.sum((self.cluster_centers_ - X)** 2, axis=0))

            # Assign labels based on the closest centroid
            self.labels_ = np.argmin(distances, axis=0)

            # Update centroids based on the mean of points in each cluster
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(self.cluster_centers_, new_centers):
                break

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
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # Assign -1 as noise points
        cluster_label = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue

            neighbors = self._find_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                cluster_label += 1
                self.labels_[i] = cluster_label
                self._expand_cluster(X, i, neighbors, cluster_label)

        return self

    def _find_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_label):
        queue = list(neighbors)

        while queue:
            current_object_idx = queue.pop(0)
            if self.labels_[current_object_idx] == -1:
                self.labels_[current_object_idx] = cluster_label

            current_neighbors = self._find_neighbors(X, current_object_idx)
            if len(current_neighbors) >= self.min_samples:
                for neighbor_idx in current_neighbors:
                    if self.labels_[neighbor_idx] in [-1, 0]:
                        queue.append(neighbor_idx)
                        self.labels_[neighbor_idx] = cluster_label

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
