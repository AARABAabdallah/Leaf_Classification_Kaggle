from sklearn.linear_model import LogisticRegression

class LogisticalRegressionModel:
    def train_model(self, x_train, y_train):
        clf = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        clf.fit(x_train,y_train)
        return clf
