import pandas as pd
from math import sqrt
from collections import Counter

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
            distances = {}
            for data, target_name in zip(self.data, self.target_names):
                distance = self.euclidean_distance(data, test_instance, len(test_instance))
                distances[distance] = target_name

            keys = list(distances.keys())
            keys.sort()
            neighbours = [distances[key] for key in keys[:self.K]]
            intermediate.append((test_instance, neighbours))
        
        for test_instance, neighbours in intermediate:
            counter = Counter(neighbours)
            result.append([test_instance, counter.most_common(1)[0][0]])
        
        return result

    def euclidean_distance(self, a, b, length):
        distance = 0
        for i in range(length):
            distance += pow(a[i] - b[i], 2)
        return sqrt(distance)
