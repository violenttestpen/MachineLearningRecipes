from StatisticalModel import covariance, mean, stdev, variance

class LinearRegression:
    coef_ = None
    intercept_ = None
    r_ = None
    rsq_ = None
    
    def fit(self, x_train, y_train):
        if len(x_train) != len(y_train):
            return None
        
        self.coef_ = covariance(x_train, y_train) / variance(x_train)
        self.intercept_ = mean(y_train) - (self.coef_ * mean(x_train))
        self.r_ = covariance(x_train, y_train) / (stdev(x_train) * stdev(y_train))
        self.rsq_ = self.r_ ** 2
        return self

    def predict(self, x_test):
        x_test = [ x_test ] if type(x_test) != list else x_test
        return list(map(lambda x: (self.coef_ * x) + self.intercept_, x_test))