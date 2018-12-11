from StatisticalModel import covariance, mean, stdev, variance


class LinearRegression:
    coef_ = None
    intercept_ = None
    r_ = None
    rsq_ = None

    def fit(self, x_train, y_train):
        """Calculates the coefficient and intercept for `y = mx + b`"""
        if len(x_train) != len(y_train):
            return None

        # m = covariance(x, y) / variance(x)
        self.coef_ = covariance(x_train, y_train) / variance(x_train)

        # b = y - mx
        self.intercept_ = mean(y_train) - (self.coef_ * mean(x_train))

        # pearson's r = covariance(x, y) / ( stdev(x) * stdev(y) )
        self.r_ = covariance(x_train, y_train) / (stdev(x_train) * stdev(y_train))

        # coefficient of determination = (pearson's r) ^ 2
        self.rsq_ = self.r_ ** 2
        return self

    def predict(self, x_test):
        """Predicts the y based on x using the formula `y = mx + b`"""
        x_test = [x_test] if type(x_test) != list else x_test
        return list(map(lambda x: (self.coef_ * x) + self.intercept_, x_test))
