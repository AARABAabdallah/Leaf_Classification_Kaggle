import data.data_manipulation as dm
from sklearn import svm
import pickle
from sklearn import metrics

class SvmModel:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.clf = None
        self.data_manip = dm.DataManipulation()

    def load_data(self):
        self.x_train,self.y_train = self.data_manip.get_training_data()
        self.x_test,self.y_test = self.data_manip.get_test_data()

    def train_model(self,kernel = 'linear',degree = 3,gamma = 'auto'):
        """The function that serves to create the SVM classifier (SVC), and train it with
        the training data already set in instance attribute self.x_train and its labels self.y_train.
        Note : The kernel functions that can be used are : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’ and ‘precomputed’
        hyper parameters can be set using args : degree ( of polynomial function, by default 3),
        and gamma ( Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’, by default 1 / n_features )
        """
        #loading the data
        self.load_data()
        # Create a svm Classifier
        self.clf = svm.SVC(kernel,probability=True,degree=degree,gamma=gamma)
        # Train the model using the training sets
        self.clf.fit(self.x_train, self.y_train)

    def predict_labels(self,x=None):
        if x is None:
            x = self.x_test
        # Predict the response for test dataset
        return self.clf.predict(x)

    def calcul_test_accuracy(self):
        self.test_accracy = self.clf.score(self.x_test,self.y_test)

    def save_model(self,filename='svm_model'):
        pickle.dump(self.clf, open("../models/"+filename, 'wb'))

    def load_model(self,filename='svm_model'):
        self.clf = pickle.load(open("../models/"+filename, 'rb'))