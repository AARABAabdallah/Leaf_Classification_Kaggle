from sklearn.linear_model import LogisticRegression
import data.data_manipulation as dm
from sklearn.metrics import log_loss

class LogisticalRegressionModel:
    def __init__(self):
        self.clf_all_data = None
        self.clf_splited_data = None
        self.data = None
        self.labels = None
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None
        self.data_unlabeled = None
        self.training_loss_all_data = None
        self.training_loss_splited_data = None
        self.test_loss_data_splited = None
        self.data_man = dm.DataManipulation()
        self.probas_test_data_splited = None
        self.probas_train_data_splited = None
        self.probas_all_train_data = None # 2D array containing in each i and each column j
                                 # the probability that x[i] belongs to the class j
                                 # where each class j corresponds to its order within self.clf.classes

    def train_model(self, type='all_data'):
        if type == 'all_data':
            self.load_data('all_data')
            self.clf_all_data = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_all_data.fit(self.data, self.labels)
        else:
            self.load_data('all_data')
            self.clf_splited_data = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_splited_data.fit(self.data, self.labels)

    def load_data(self, type='data_splited'):
        self.data_man.load_data()
        if type == 'all_data':
            self.data, self.labels = self.data_man.get_data()
        else:
            self.data_train, self.labels_train = self.data_man.get_training_data()

    def predict_proba_train(self, type='data_splited'):
        if type == 'all_data':
            self.probas_all_train_data = self.clf_all_data.predict_proba(self.data)
        else:
            self.probas_train_data_splited = self.clf_splited_data.predict_proba(self.data_train)

    def predict_proba_test_splited_data(self):
        self.probas_test_data_splited = self.clf_splited_data.predict_proba(self.data_test)

    def calculate_training_loss(self, type='data_splited'):
        if type == 'all_data':
            self.predict_proba_train('all_data')
            self.training_loss_all_data = log_loss(y_true=self.labels, y_pred=self.probas_all_train_data,
                                                   labels=self.clf_all_data.classes_)
            return self.training_loss_all_data
        else:
            self.predict_proba_train('data_splited')
            self.training_loss_all_data = log_loss(y_true=self.labels_train, y_pred=self.probas_train_data_splited,
                                                   labels=self.clf_splited_data.classes_)
            return self.training_loss_splited_data

    def calculate_test_loss_splited_data(self):
        self.predict_proba_test_splited_data()
        self.test_loss_data_splited = log_loss(y_true=self.labels_test, y_pred=self.probas_test_data_splited,
                                                   labels=self.clf_splited_data.classes_)
        return self.test_loss_data_splited

    def load_unlabeled_data(self):
        self.data_unlabeled = self.data_man.get_unlabeled_data()
