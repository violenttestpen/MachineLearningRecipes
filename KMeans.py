from random import randint


class KMeans:
    # cluster_centers_ : array, [n_clusters, n_features]
    clusters_centers_ = []
    labels_ = []
    inertia_ = 0.0

    def __init__(self, n_clusters=8, max_iter=300, tol=.0001, n_features=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_features = n_features

    def fit(self, X):
        # http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
        # Initialize centroids randomly
        centroids = [list(map(lambda x: randint(0, 10), [1] * self.n_features))
                     for i in range(self.n_clusters)]  # random

        # Initialize book keeping vars.
        old_centroids = None
        n_iter = 0

        # Run the main k-means algorithm
        while (n_iter <= self.max_iter) and (old_centroids != centroids):
            # Save old centroids for convergence test. Book keeping.
            old_centroids = centroids
            n_iter += 1

            # Assign labels to each datapoint based on centroids
            for datapoint in X:
                pass

            # Assign centroids based on datapoint labels
        # We can get the labels too by calling getLabels(dataSet, centroids)
        return self
