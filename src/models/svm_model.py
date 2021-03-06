import data.data_manipulation as dm
from sklearn import svm
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
import os
from sklearn.metrics import log_loss
import pandas as pd
import enlighten
import tqdm

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

    def train_model(self,kernel = 'linear',degree = 3,gamma = 'auto',coef0=0,all_data = True):
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
        self.save_model()


    def predict_labels(self,x=None):
        if x is None:
            x = self.x_test
        # Predict the response for test dataset
        return self.clf.predict(x)

    def calcul_test_accuracy(self):
        self.test_accuracy = self.clf.score(self.x_test,self.y_test)

    def calculate_training_loss(self):
        data, labels = self.data_manip.get_data()
        probas_all_train_data = self.clf.predict_proba(data)
        training_loss_all_data = log_loss(y_true=labels, y_pred=probas_all_train_data,
                                                   labels=self.clf.classes_)
        return training_loss_all_data

    def submit_test_results_features(self,kernel='default'):
        # Here we write the submission.csv file according to this model to be submitted in kaggle
        data_unlabeled = self.data_manip.get_unlabeled_data()
        if kernel == 'rbf':
            self.clf_rbf = self.clf
            probas_unlabeled = self.clf_rbf.predict_proba(data_unlabeled)
            header = self.clf_rbf.classes_
            df = pd.DataFrame(probas_unlabeled, columns=header)
            test_ids = self.data_manip.get_test_data_ids()
            df.insert(loc=0, column='id', value=test_ids)
            df.to_csv(r'../data_sets/submissions/svm_rbf_test_results.csv', index=None)
        elif kernel == 'sigmoid':
            self.clf_sigmoid = self.clf
            probas_unlabeled = self.clf_sigmoid.predict_proba(data_unlabeled)
            header = self.clf_sigmoid.classes_
            df = pd.DataFrame(probas_unlabeled, columns=header)
            test_ids = self.data_manip.get_test_data_ids()
            df.insert(loc=0, column='id', value=test_ids)
            df.to_csv(r'../data_sets/submissions/svm_sigmoid_test_results.csv', index=None)
        elif kernel == 'poly':
            self.clf_poly = self.clf
            probas_unlabeled = self.clf_poly.predict_proba(data_unlabeled)
            header = self.clf_poly.classes_
            df = pd.DataFrame(probas_unlabeled, columns=header)
            test_ids = self.data_manip.get_test_data_ids()
            df.insert(loc=0, column='id', value=test_ids)
            df.to_csv(r'../data_sets/submissions/svm_poly_test_results.csv', index=None)
        else:
            probas_unlabeled = self.clf.predict_proba(data_unlabeled)
            header = self.clf.classes_
            df = pd.DataFrame(probas_unlabeled, columns=header)
            test_ids = self.data_manip.get_test_data_ids()
            df.insert(loc=0, column='id', value=test_ids)
            df.to_csv(r'../data_sets/submissions/svm_test_results.csv', index=None)


    def save_model(self,filename='svm_model',kernel='default'):
        if kernel == 'default':
            joblib.dump(self.clf, "../models/"+filename + '.joblib')
        elif kernel == 'rbf':
            joblib.dump(self.clf_rbf, "../models/" + filename + '_rbf.joblib')
        elif kernel == 'sigmoid':
            joblib.dump(self.clf_sigmoid, "../models/" + filename + '_sigmoid.joblib')
        elif kernel == 'poly':
            joblib.dump(self.clf_poly, "../models/" + filename + '_poly.joblib')
        else:
            print("The specified kernel is not an option. Please choose one of the following options : poly OR rbf OR sigmoid.")

    def load_model(self,filename='svm_model',kernel='default'):
        if kernel == 'default':
            self.clf = joblib.load("../models/"+filename + '.joblib')
        elif kernel == 'rbf':
            self.clf_rbf = joblib.load("../models/" + filename + '_rbf.joblib')
        elif kernel == 'sigmoid':
            self.clf_sigmoid = joblib.load("../models/" + filename + '_sigmoid.joblib')
        elif kernel == 'poly':
            self.clf_poly = joblib.load("../models/" + filename + '_poly.joblib')

    def auto_load_models(self,filename='svm_model'):
        if os.path.isfile("../models/"+filename + '.joblib'):
            self.load_model()
        if os.path.isfile("../models/"+filename + '_rbf.joblib'):
            self.load_model(kernel='rbf')
            if self.clf == None:
                self.clf = self.clf_rbf
        if os.path.isfile("../models/"+filename + '_sigmoid.joblib'):
            self.load_model(kernel='sigmoid')
            if self.clf == None:
                self.clf = self.clf_sigmoid
        if os.path.isfile("../models/"+filename + '_poly.joblib'):
            self.load_model(kernel='poly')
            if self.clf == None:
                self.clf = self.clf_poly

    def cross_validation(self,kernel = 'rbf'):
        #kernels = ['poly', 'rbf', 'sigmoid'] # we don't consider the linear kernel because it is the same as poly kernel with degree = 1
        if kernel == 'rbf':
            # Pour afficher l'avancement des coucles
            pbar = enlighten.Counter(total=(len([0.000001*(10**i) for i in range(6)]+list(np.arange(0.1, 2.1, 0.1))+[2**i for i in range(1,10)]+['auto'])), desc='Basic', unit='ticks')
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
                pbar.update()  # mise à jour de l'avancement des boucles
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
            gamma_list = list(np.arange(0.1, 2.1, 0.2)) + [2 ** i for i in range(1, 6)] + ['auto']
            # gamma_list =  ['auto']
            coef0_list = [0.001 * (10 ** i) for i in range(3)]+[2 ** i for i in range(6)]
            coef0_list =[2 ** i for i in range(2,6)]
            pbar = enlighten.Counter(total=(len(gamma_list) * len(coef0_list) ), desc='Basic',unit='ticks')
            for gamma in gamma_list:
                for coef0 in coef0_list :
                    self.train_model(kernel='sigmoid', gamma=gamma, coef0=coef0)
                    mean_score = cross_val_score(self.clf, self.x_train, self.y_train, cv=5).mean()
                    if mean_score > best_score :
                        best_score = mean_score
                        best_gamma = gamma
                        best_coef0 = coef0
                        print('gama = ',best_gamma)
                        print('coef0 = ',best_coef0)
                        print('score = ',best_score,end='\n\n')
                    pbar.update()
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
            #gamma_list =list(np.arange(0.1, 2.1, 0.3)) + [2 ** i for i in range(1, 5)] + ['auto']
            gamma_list = [2 ** i for i in range(1, 5)] + ['auto']
            coef0_list = [2 ** i for i in range(2,6)]
            degree_list = range(1,6)
            pbar = enlighten.Counter(total=(len(gamma_list)*len(coef0_list)*len(degree_list)),desc='Basic', unit='ticks')
            for gamma in gamma_list:
                for coef0 in coef0_list :
                    for degree in degree_list:
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
                        pbar.update()  # mise à jour de l'avancement des boucles
            self.poly_gamma = best_gamma
            self.poly_coef0 = best_coef0
            self.poly_degree = best_degree
            self.train_model(kernel='poly',gamma=self.poly_gamma,coef0=self.poly_coef0,degree=self.poly_degree)
            self.clf_poly = self.clf
            self.save_model(kernel='poly')
            print("resulting gamma : ",self.poly_gamma)
            print("resulting coef0 : ",self.poly_coef0)
            print("resulting degree : ",self.poly_degree)
            print()
        else:
            print("The specified kernel is not an option. Please choose one of the following options : poly OR rbf OR sigmoid.")
            print("By default the kernel will be 'rbf'")

    ############################# model based on PCA transformed data #############################
    def train_model_pca_cross_validation(self, type='all_data', kernel='linear', degree=3, gamma='auto', coef0=0):
        # the validation_loss is calculated according to the type:
        # if type == 'data_splited' then the training and the validassion loss will be done according to the splited data
        # else(type = 'all_data' then the training and the validassion loss will be done according to all the data
        err_val_min = 100
        nbr_compoenents_min = 1
        for i in tqdm.tqdm(range(2, 192)):
            if type == 'data_splited':
                self.train_model_pca('data_splited', num_comp=i, kernel=kernel,
                                     degree=degree, gamma=gamma, coef0=coef0)
                validation_loss = self.calculate_validation_loss_pca_data()
            else:
                # save=False so that we don't save the model in each iteration
                self.train_model_pca('all_data', num_comp=i, save=False, kernel=kernel,
                                     degree=degree, gamma=gamma, coef0=coef0)
                validation_loss = self.calculate_training_loss_pca_data()
                
            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i

        print("nbr_comp_min: ", nbr_compoenents_min)
        self.train_model_pca('all_data', num_comp=nbr_compoenents_min, kernel=kernel,
                             degree=degree, gamma=gamma, coef0=coef0)
        return nbr_compoenents_min

    def train_model_pca(self, type='all_data', num_comp=167, save=True, kernel='linear', degree=3, gamma='auto', coef0=0):
        # type is a parameter that specifies wether we train the model on all the training data,
            # or only on the splited data training, will be needed in the cross validation function
        # num_comp: is a parameter that specifies the number the components to hold after the data transformation
        # save: is a parameter that specifies wether to save the model or not (only when type!='all_data')
        self.data_manip.load_pca_data(num_components=num_comp)
        if type == 'data_splited':
            data_pca_transformed_splited_train = self.data_manip.get_data_pca_transformed_splited_train()
            labels_train = self.data_manip.get_labels_splited_train()
            self.clf_pca_model_splited = svm.SVC(
                kernel=kernel, probability=True, degree=degree, gamma=gamma, coef0=coef0)
            self.clf_pca_model_splited.fit(
                data_pca_transformed_splited_train, labels_train)
        else:
            data_pca_transformed = self.data_manip.get_data_pca_transformed()
            labels = self.data_manip.get_labels()
            self.clf_pca_model = svm.SVC(kernel=kernel, probability=True, degree=degree, gamma=gamma, coef0=coef0)
            self.clf_pca_model.fit(data_pca_transformed, labels)
            if save == True:
                self.save_model_pca(num_comp=num_comp,kernel=kernel)
                self.nbr_comp_pca_model_trained_with = num_comp

    def calculate_training_loss_pca_data(self):
        labels = self.data_manip.get_labels()
        data_pca_transformed = self.data_manip.get_data_pca_transformed()
        probas_pca_data = self.clf_pca_model.predict_proba(
            data_pca_transformed)
        training_loss_pca_data = log_loss(y_true=labels, y_pred=probas_pca_data,
                                          labels=self.clf_pca_model.classes_)
        return training_loss_pca_data

    def calculate_validation_loss_pca_data(self):
        labels_test = self.data_manip.get_labels_splited_test()
        data_pca_transformed_splited_test = self.data_manip.get_data_pca_transformed_splited_test()
        probas_pca_data_test = self.clf_pca_model_splited.predict_proba(
            data_pca_transformed_splited_test)
        self.training_loss_pca_data_test = log_loss(y_true=labels_test, y_pred=probas_pca_data_test,
                                                    labels=self.clf_pca_model_splited.classes_)
        return self.training_loss_pca_data_test

    def save_model_pca(self, num_comp=160,  kernel='default'):
        joblib.dump(self.clf_pca_model,
                    '../models/svm_pca_model_'+kernel+'_'+str(num_comp)+'.joblib')

    def load_model_pca(self, num_comp=160, kernel = 'default'):
        self.data_manip.load_pca_data(num_components=num_comp)
        self.nbr_comp_pca_model_trained_with = num_comp
        self.clf_pca_model = joblib.load(
            '../models/svm_pca_model_'+kernel+'_'+str(num_comp)+'.joblib')

    def submit_test_results_pca(self):
        data_unlabeled_pca_transformed = self.data_manip.get_unlabeled_pca_transformed()
        probas_unlabeled_pca = self.clf_pca_model.predict_proba(
            data_unlabeled_pca_transformed)
        header = self.clf_pca_model.classes_
        df = pd.DataFrame(probas_unlabeled_pca, columns=header)
        test_ids = self.data_manip.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/svm_test_results_pca_nbcomp_' +
                  str(self.nbr_comp_pca_model_trained_with)+'.csv', index=None)
