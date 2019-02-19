from collections import Counter


class Discriminant:
    column = None
    value = None

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, data):
        val = data[self.column]
        return (val >= self.value) if val.isnumeric() else (val == self.value)

    def __repr__(self):
        condition = '==' if type(self.value) in (int, float) else '>='
        return f"Is {header[self.column]} {condition} {self.value}?"


def partition(rows, discriminant):
    result = {True: [], False: []}
    for row in rows:
        result[discriminant.match(row)].append(row)
    return result[True], result[False]


def gini(rows, label_column=-1):
    counter = Counter(map(lambda x: x[label_column], rows))
    impurity = 1
    for label in counter.keys():
        impurity -= pow(counter[label] / float(len(rows)), 2)
    return impurity


class DecisionTreeClassifier:
    pass


training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ['Color', 'Diameter', 'Label']
print(Discriminant(1, 3))
print(Discriminant(0, 'Green'))

true_rows, false_rows = partition(training_data, Discriminant(0, 'Red'))
print(true_rows)
print(false_rows)
print(gini([['Apple'], ['Orange']]))
