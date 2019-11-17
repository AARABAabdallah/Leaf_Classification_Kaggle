from sklearn.linear_model import LogisticRegression
import data.data_manipulation as dm
from sklearn.metrics import log_loss

class LogisticalRegressionModel:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.clf = None
        self.data_man = dm.DataManipulation()
        self.probas_train = None # 2D array containing in each i and each column j
                                 # the probability that x[i] belongs to the class j
                                 # where each class j corresponds to its order within self.clf.classes

    def train_model(self):
        self.clf = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        #self.x_train, self.y_train = self.data_man.load_data_train()
        self.load_data()
        self.clf.fit(self.x_train, self.y_train)
        #return clf

    def load_data(self):
        self.x_train, self.y_train = self.data_man.load_data_train()
        self.x_test = self.data_man.load_data_test()

    def predict_proba_train(self):
        self.probas_train = self.clf.predict_proba(self.x_train)

    def calculate_loss(self):
        self.predict_proba_train()
        self.training_loss = log_loss(y_true=self.y_train, y_pred=self.probas_train, labels=self.clf.classes_)
        return self.training_loss
    