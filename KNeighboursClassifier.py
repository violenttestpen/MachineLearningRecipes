from collections import Counter
from StatisticalModel import euclidean_distance_2d


class KNeightboursClassifier:
    K = 5
    data = []
    target_names = []

    def __init__(self, n_neighbours=5):
        self.K = n_neighbours

    def fit(self, X, y):
        self.data = X
        self.target_names = y

    def predict(self, X):
        result = []
        for instance in X:
            # Calculate the Euclidean distance of each data point
            # A distance keypair is in the format `euclidean_distance: label`
            distances = {
                euclidean_distance_2d(data, instance, len(instance)): label
                for data, label in zip(self.data, self.target_names)
            }

            # Retrieve the top-K nearest neighbour and store it in an
            # intermediary result (instance: neighbours)
            keys = sorted(list(distances.keys()))
            neighbours = [distances[key] for key in keys[:self.K]]

            # Classify test instance with the most common neighbour
            counter = Counter(neighbours)
            result.append([instance, counter.most_common(1)[0][0]])

        return result
