import data.data_manipulation as dm
from sklearn import svm
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
import os

class SvmModel:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_all_data = None
        self.y_all_data = None
        self.clf = None
        self.test_accuracy = None
        self.data_manip = dm.DataManipulation()
        self.load_data()
        #hyper parameters
        self.rbf_gamma = None
        self.sigmoid_coef0 = None
        self.sigmoid_gamma = None
        self.poly_degree = None
        self.poly_gamma = None
        self.poly_coef0 = None
        #classifier for every kernel after cross-validation
        self.clf_rbf = None
        self.clf_sigmoid = None
        self.clf_poly = None
        #load models if they was already stored
        self.auto_load_models()



    def load_data(self):
        self.x_train,self.y_train = self.data_manip.get_training_data()
        self.x_test,self.y_test = self.data_manip.get_test_data()
        self.x_all_data,self.y_all_data = self.data_manip.get_data()

    def train_model(self,kernel = 'linear',degree = 3,gamma = 'auto',coef0=0,all_data = False):
        """The function that serves to create the SVM classifier (SVC), and train it with
        the training data already set in instance attribute self.x_train and its labels self.y_train.
        Note : The kernel functions that can be used are : ‘linear’, ‘poly’, ‘rbf’ or ‘sigmoid’
        hyper parameters can be set using args :
            degree : of polynomial function, by default 3
            gamma : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’, by default 1 / n_features
            coef0 : The constant that figures in 'poly' and 'sigmoid' kernels

        """
        if all_data:
            train_data,train_label = self.x_all_data,self.y_all_data
        else:
            train_data, train_label = self.x_train,self.y_train
        # Create a svm Classifier
        self.clf = svm.SVC(kernel=kernel,probability=True,degree=degree,gamma=gamma,coef0=coef0)
        # Train the model using the training sets
        self.clf.fit(train_data, train_label)

    def predict_labels(self,x=None):
        if x is None:
            x = self.x_test
        # Predict the response for test dataset
        return self.clf.predict(x)

    def calcul_test_accuracy(self):
        self.test_accuracy = self.clf.score(self.x_test,self.y_test)

    def save_model(self,filename='svm_model',kernel='default'):
        if kernel == 'default':
            joblib.dump(self.clf, open("../models/"+filename + '.joblib', 'wb'))
        elif kernel == 'rbf':
            joblib.dump(self.clf_rbf, open("../models/" + filename + '_rbf.joblib', 'wb'))
        elif kernel == 'sigmoid':
            joblib.dump(self.clf_sigmoid, open("../models/" + filename + '_sigmoid.joblib', 'wb'))
        elif kernel == 'poly :':
            joblib.dump(self.clf_poly, open("../models/" + filename + '_poly.joblib', 'wb'))
        else:
            print("The specified kernel is not an option. Please choose one of the following options : poly OR rbf OR sigmoid.")

    def load_model(self,filename='svm_model',kernel='default'):
        if kernel == 'default':
            self.clf = joblib.load(open("../models/"+filename + '.joblib', 'rb'))
        elif kernel == 'rbf':
            self.clf_rbf = joblib.load(open("../models/" + filename + '_rbf.joblib', 'wb'))
        elif kernel == 'sigmoid':
            self.clf_sigmoid = joblib.load(open("../models/" + filename + '_sigmoid.joblib', 'wb'))
        elif kernel == 'poly :':
            self.clf_poly = joblib.load(open("../models/" + filename + '_poly.joblib', 'wb'))

    def auto_load_models(self,filename='svm_model'):
        if os.path.isfile("../models/"+filename + '.joblib'):
            self.load_model()
        if os.path.isfile("../models/"+filename + '_rbf.joblib'):
            self.load_model(kernel='rbf')
        if os.path.isfile("../models/"+filename + '_sigmoid.joblib'):
            self.load_model(kernel='sigmoid')
        if os.path.isfile("../models/"+filename + '_poly.joblib'):
            self.load_model(kernel='poly')

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
            self.clf_rbf = self.clf
            self.save_model(kernel='rbf')
            print("resulting gamma : ",self.rbf_gamma)

        elif kernel == 'sigmoid':
            self.train_model(kernel='sigmoid',gamma=0.1,coef0=0)
            best_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
            best_gamma = 0.1
            best_coef0 = 0
            for gamma in list(np.arange(0.1, 2.1, 0.2)) + [2 ** i for i in range(1, 6)] + ['auto']:
                for coef0 in [0.001 * (10 ** i) for i in range(3)]+[2 ** i for i in range(6)] :
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
            self.clf_sigmoid = self.clf
            self.save_model(kernel='sigmoid')
            print("resulting gamma : ",self.sigmoid_gamma)
            print("resulting coef0 : ",self.sigmoid_coef0)

        elif kernel == 'poly':
            self.train_model(kernel='poly',gamma=0.1,coef0=0,degree=1)
            best_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
            best_gamma = 0.1
            best_coef0 = 0
            best_degree = 1
            for gamma in list(np.arange(0.1, 2.1, 0.3)) + [2 ** i for i in range(1, 5)] + ['auto']:
                for coef0 in [2 ** i for i in range(6)] :
                    for degree in range(1,6):
                        self.train_model(kernel='poly', gamma=gamma, coef0=coef0,degree=degree)
                        mean_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
                        if mean_score > best_score :
                            best_score = mean_score
                            best_gamma = gamma
                            best_coef0 = coef0
                            best_degree = degree
                            print('gama = ',best_gamma)
                            print('coef0 = ',best_coef0)
                            print('degree = ',best_degree)
                            print('score = ',best_score,end='\n\n')
            self.poly_gamma = best_gamma
            self.poly_coef0 = best_coef0
            self.poly_degree = best_degree
            self.train_model(kernel='poly',gamma=self.poly_gamma,coef0=self.poly_coef0,degree=self.poly_degree)
            self.clf_poly = self.clf
            self.save_model(kernel='poly')
            print("resulting gamma : ",self.poly_gamma)
            print("resulting coef0 : ",self.poly_coef0)
            print("resulting degree : ",self.poly_degree)
        else:
            print("The specified kernel is not an option. Please choose one of the following options : poly OR rbf OR sigmoid.")
            print("By default the kernel will be 'rbf'")


