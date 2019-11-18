import data.data_manipulation as dm
from sklearn import svm
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics

class SvmModel:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.clf = None
        self.test_accuracy = None
        self.data_manip = dm.DataManipulation()
        self.data_manip.load_data()
        self.data_manip.split_data()
        self.load_data()
        #hyper parameters
        self.rbf_gamma = None



    def load_data(self):
        self.x_train,self.y_train = self.data_manip.get_training_data()
        self.x_test,self.y_test = self.data_manip.get_test_data()

    def train_model(self,kernel = 'linear',degree = 3,gamma = 'auto',coef0=0):
        """The function that serves to create the SVM classifier (SVC), and train it with
        the training data already set in instance attribute self.x_train and its labels self.y_train.
        Note : The kernel functions that can be used are : ‘linear’, ‘poly’, ‘rbf’ or ‘sigmoid’
        hyper parameters can be set using args :
            degree : of polynomial function, by default 3
            gamma : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’, by default 1 / n_features
            coef0 : The constant that figures in 'poly' and 'sigmoid' kernels

        """
        # Create a svm Classifier
        self.clf = svm.SVC(kernel=kernel,probability=True,degree=degree,gamma=gamma,coef0=coef0)
        # Train the model using the training sets
        self.clf.fit(self.x_train, self.y_train)

    def predict_labels(self,x=None):
        if x is None:
            x = self.x_test
        # Predict the response for test dataset
        return self.clf.predict(x)

    def calcul_test_accuracy(self):
        self.test_accuracy = self.clf.score(self.x_test,self.y_test)

    def save_model(self,filename='svm_model'):
        pickle.dump(self.clf, open("../models/"+filename, 'wb'))

    def load_model(self,filename='svm_model'):
        self.clf = pickle.load(open("../models/"+filename, 'rb'))

    def cross_validation(self,kernel = 'rbf'):
        #kernels = ['poly', 'rbf', 'sigmoid'] # we don't consider the linear kernel because it is the same as poly kernel with degree = 1
        if kernel == 'rbf':
            self.train_model(kernel='rbf',gamma=0.001)
            best_gamma_score = cross_val_score(self.clf,self.x_train,self.y_train,cv=5).mean()
            best_gamma = 0.001
            for gamma in [0.000001*(10**i) for i in range(6)]+list(np.arange(0.1, 2.1, 0.1))+[2**i for i in range(1,10)]+['auto']:
                self.train_model(kernel='rbf',gamma=gamma)
                mean_score = cross_val_score(self.clf,self.x_train,self.y_train,cv=5).mean()
                print(mean_score,gamma)
                if mean_score > best_gamma_score :
                    best_gamma_score = mean_score
                    best_gamma = gamma
                    print('gama = ',best_gamma)
                    print('score = ',best_gamma_score,end='\n\n')
            self.rbf_gamma = best_gamma
            self.train_model(kernel='rbf',gamma=self.rbf_gamma)
            print("resulting gamma : ",self.rbf_gamma)

        elif kernel == 'sigmoid':
            self.train_model(kernel='sigmoid',gamma=0.1,coef0=0)
            best_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
            best_gamma = 0.1
            best_coef0 = 0
            for gamma in list(np.arange(0.1, 2.1, 0.1)) + [2 ** i for i in range(1, 10)] + ['auto']:
                for coef0 in [0.001 * (10 ** i) for i in range(3)]+[2 ** i for i in range(10)] :
                    self.train_model(kernel='sigmoid', gamma=gamma, coef0=coef0)
                    mean_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
                    if mean_score > best_score :
                        best_score = mean_score
                        best_gamma = gamma
                        best_coef0 = coef0
                        print('gama = ',best_gamma)
                        print('coef0 = ',best_coef0)
                        print('score = ',best_score,end='\n\n')
            self.sigmoid_gamma = best_gamma
            self.sigmoid_coef0 = best_coef0
            self.train_model(kernel='sigmoid',gamma=self.sigmoid_gamma,coef0=self.sigmoid_coef0)
            print("resulting gamma : ",self.sigmoid_gamma)
            print("resulting coef0 : ",self.sigmoid_coef0)
        elif kernel == 'poly':
            self.train_model(kernel='poly',gamma=0.1,coef0=0,degree=1)
            best_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
            best_gamma = 0.1
            best_coef0 = 0
            best_degree = 1
            for gamma in list(np.arange(0.1, 2.1, 0.1)) + [2 ** i for i in range(1, 5)] + ['auto']:
                for coef0 in [2 ** i for i in range(10)] :
                    for degree in range(2,10):
                        self.train_model(kernel='sigmoid', gamma=gamma, coef0=coef0)
                        mean_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
                        if mean_score > best_score :
                            best_score = mean_score
                            best_gamma = gamma
                            best_coef0 = coef0
                            print('gama = ',best_gamma)
                            print('coef0 = ',best_coef0)
                            print('score = ',best_score,end='\n\n')
            self.sigmoid_gamma = best_gamma
            self.sigmoid_coef0 = best_coef0
            self.train_model(kernel='sigmoid',gamma=self.sigmoid_gamma,coef0=self.sigmoid_coef0)
            print("resulting gamma : ",self.sigmoid_gamma)
            print("resulting coef0 : ",self.sigmoid_coef0)



