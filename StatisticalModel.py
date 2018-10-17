from math import sqrt

def mean(data):
    return sum(data) / float(len(data))

def variance(data, type='sample'):
    mu = mean(data)
    n = (len(data) - 1) if 'sample'.startswith(type) else len(data)
    return sum([(x - mu)**2 for x in data]) / float(n)

def stdev(data, type='sample'):
    return sqrt(variance(data, type))

def covariance(x_data, y_data, type='sample'):
    n = (len(x_data) - 1) if 'sample'.startswith(type) else len(x_data)
    x_mean, y_mean = mean(x_data), mean(y_data)
    return sum([ (x - x_mean) * (y - y_mean) for x, y in zip(x_data, y_data) ]) / float(n)