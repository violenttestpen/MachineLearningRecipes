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
        result, intermediate = [], []
        for test_instance in X:
            # Calculate the Euclidean distance of each data point with the test instance
            # A distance keypair is in the format `euclidean_distance: label`
            distances = {}
            for data, target_name in zip(self.data, self.target_names):
                distance = euclidean_distance_2d(data, test_instance, len(test_instance))
                distances[distance] = target_name

            # Retrieve the top-K nearest neighbour and store it in an intermediary result
            keys = sorted(list(distances.keys()))
            neighbours = [distances[key] for key in keys[:self.K]]
            intermediate.append((test_instance, neighbours))

        # Classify test instance with the most common Label of the neighbours
        for test_instance, neighbours in intermediate:
            counter = Counter(neighbours)
            result.append([test_instance, counter.most_common(1)[0][0]])

        return result
