from math import sqrt


def mean(data):
    return sum(data) / float(len(data))


def median(data):
    length = len(data)
    result = data[length // 2]
    return mean([result, data[length // 2 - 1]]) if length % 2 == 0 else result


def variance(data, mode='sample'):
    mu = mean(data)
    n = (len(data) - 1) if 'sample'.startswith(mode) else len(data)
    return sum([(x - mu)**2 for x in data]) / float(n)


def stdev(data, mode='sample'):
    return sqrt(variance(data, mode))


def covariance(x_data, y_data, mode='sample'):
    n = (len(x_data) - 1) if 'sample'.startswith(mode) else len(x_data)
    x_mean, y_mean = mean(x_data), mean(y_data)
    return sum([(x - x_mean) * (y - y_mean) for x, y in zip(x_data, y_data)]) / float(n)


def euclidean_distance_2d(a, b, length):
    distance = 0
    for i in range(length):
        distance += pow(a[i] - b[i], 2)
    return sqrt(distance)


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def levensthein_distance(s, t):
    # base case: empty strings
    len_s, len_t = len(s), len(t)
    if len_s == 0:
        return len_t
    if len_t == 0:
        return len_s

    # test if last characters of the strings match
    cost = 0 if s[-1] == t[-1] else 1

    # return minimum of delete char from s,
    # delete char from t, and delete char from both
    sub_s, sub_t = s[:-1], t[:-1]
    return min(levensthein_distance(sub_s, t) + 1,
               levensthein_distance(s, sub_t) + 1,
               levensthein_distance(sub_s, sub_t) + cost)
