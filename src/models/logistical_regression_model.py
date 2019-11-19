from sklearn.linear_model import LogisticRegression
import data.data_manipulation as dm
from sklearn.metrics import log_loss
import joblib
import pandas as pd
import numpy as np
import tqdm as tqdm


class LogisticalRegressionModel:
    def __init__(self):
        self.clf_all_data = None
        #self.clf_splited_data = None
        self.clf_images_model = None
        self.clf_images_features_model = None
        self.clf_pca_model_splited = None

        self.data_pca_transformed_splited_train = None
        self.data_pca_transformed_splited_test = None
        self.data = None
        self.labels = None
        self.data_train = None
        self.data_test = None
        self.data_unlabeled = None
        self.features_images_train = None
        self.images_train = None
        self.images_unlabeled = None
        self.features_images_unlabeled = None

        self.labels_train = None
        self.labels_test = None

        self.data_pca_transformed = None
        self.data_unlabeled_pca_transformed = None

        self.training_loss_all_data = None
        self.training_loss_splited_data = None
        self.training_loss_data_train_images = None
        self.training_loss_data_train_features_images = None
        self.test_loss_data_splited = None
        self.training_loss_pca_data = None
        self.training_loss_pca_data_train = None
        self.training_loss_pca_data_test = None

        self.data_man = dm.DataManipulation()

        self.probas_pca_data_train = None
        self.probas_pca_data_test = None
        self.probas_unlabeled_pca = None
        self.probas_pca_data = None
        self.probas_unlabeled_features_images = None
        self.probas_data_train_features_images = None
        self.probas_data_train_images = None
        self.probas_unlabeled = None
        self.probas_unlabeled_images = None
        self.probas_test_data_splited = None
        self.probas_train_data_splited = None
        self.probas_all_train_data = None  # 2D array containing in each i and each column j
        # the probability that x[i] belongs to the class j
        # where each class j corresponds to its order within self.clf.classes
        self.nbr_comp_pca_model_trained_with = None

        self.data_man.load_data()

    ############################# Features Trained model #############################
    def train_model_features(self):
        data, labels = self.data_man.get_data()
        self.clf_all_data = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        self.clf_all_data.fit(data, labels)
        self.save_model_features()

    def calculate_training_loss(self): #type='data_splited'):
        data, labels = self.data_man.get_data()
        probas_all_train_data = self.clf_all_data.predict_proba(data)
        training_loss_all_data = log_loss(y_true=labels, y_pred=probas_all_train_data,
                                                   labels=self.clf_all_data.classes_)
        return training_loss_all_data

    def save_model_features(self):
        joblib.dump(self.clf_all_data, '../models/lregr_all_data_model.joblib')

    def load_model_features(self):
        self.clf_all_data = joblib.load('../models/lregr_all_data_model.joblib')

    def submit_test_results(self):
        data_unlabeled = self.data_man.get_unlabeled_data()
        probas_unlabeled = self.clf_all_data.predict_proba(data_unlabeled)
        header = self.clf_all_data.classes_
        df = pd.DataFrame(probas_unlabeled, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results.csv', index=None)

    ############################# Images trained model #############################
    def train_model_images(self):
        images_train = self.data_man.get_images_data_train()
        labels = self.data_man.get_labels()
        self.clf_images_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        self.clf_images_model.fit(images_train, labels)
        self.save_model_images()

    def calculate_training_loss_images(self):
        labels = self.data_man.get_labels()
        images_train = self.data_man.get_images_data_train()
        probas_data_train_images = self.clf_images_model.predict_proba(images_train)
        training_loss_data_train_images = log_loss(y_true=labels, y_pred=probas_data_train_images,
                                                        labels=self.clf_images_model.classes_)
        return training_loss_data_train_images

    def save_model_images(self):
        joblib.dump(self.clf_images_model, '../models/lregr_images_data_model.joblib')

    def load_model_images(self):
        self.clf_images_model = joblib.load('../models/lregr_images_data_model.joblib')

    def submit_test_results_images(self):
        images_unlabeled = self.data_man.get_images_data_unlabeled()
        probas_unlabeled_images = self.clf_images_model.predict_proba(images_unlabeled)
        header = self.clf_images_model.classes_
        df = pd.DataFrame(probas_unlabeled_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_images.csv', index=None)


    ############################# Features, Images concatenated trained model #############################
    def concatenate_features_images_train(self):
        data, labels = self.data_man.get_data()
        images_train = self.data_man.get_images_data_train()
        features_images_train = np.concatenate((data, images_train), axis=1)
        return features_images_train, labels

    def concatenate_features_images_test(self):
        data_test = self.data_man.get_unlabeled_data()
        images_test = self.data_man.get_images_data_unlabeled()
        features_images_test = np.concatenate((data_test, images_test), axis=1)
        return features_images_test

    def train_model_images_features(self):

        features_images_train, labels = self.concatenate_features_images_train()
        self.clf_images_features_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        self.clf_images_features_model.fit(features_images_train, labels)
        self.save_model_features_images()

    def predict_probas_unlabeled_features_images(self):
        self.probas_unlabeled_features_images = self.clf_images_features_model.predict_proba(
            self.features_images_unlabeled)

    def calculate_training_loss_features_images(self):
        features_images_train, labels = self.concatenate_features_images_train()
        probas_data_train_features_images = self.clf_images_features_model.predict_proba(
            features_images_train)
        training_loss_data_train_features_images = log_loss(y_true=labels,
                                                                 y_pred=probas_data_train_features_images,
                                                                 labels=self.clf_images_model.classes_)
        return training_loss_data_train_features_images

    def save_model_features_images(self):
        joblib.dump(self.clf_images_features_model, '../models/lregr_features_images_data_model.joblib')

    def load_model_features_images(self):
        self.clf_images_features_model = joblib.load('../models/lregr_features_images_data_model.joblib')

    def submit_test_results_images_features(self):
        #self.predict_probas_unlabeled_features_images()
        features_images_unlabeled = self.concatenate_features_images_test()
        probas_unlabeled_features_images = self.clf_images_features_model.predict_proba(
            features_images_unlabeled)
        header = self.clf_images_features_model.classes_
        df = pd.DataFrame(probas_unlabeled_features_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_images_features.csv', index=None)

    ############################# model based on PCA transformed data #############################
    """
    def load_pca_data(self, num_comp=140):
        self.nbr_comp_pca_model_trained_with = num_comp
        self.data_pca_transformed, self.data_unlabeled_pca_transformed, \
        self.data_pca_transformed_splited_train, \
        self.data_pca_transformed_splited_test = self.data_man.load_pca_data(
            num_components=num_comp)

    def train_model_pca_cross_validation_on_original_data(self):
        err_val_min = 1
        nbr_compoenents_min = 1
        for i in tqdm.tqdm(range(1, 192)):
            self.train_model_pca('all_data', num_comp=i)
            validation_loss = self.calculate_training_loss_pca_data()
            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i

        print("nbr_comp_min: ", nbr_compoenents_min)
        self.train_model_pca('all_data', num_comp=nbr_compoenents_min)
    """

    # the validation_loss is calculated according to the type
    def train_model_pca_cross_validation(self, type='data_splited'):
        err_val_min = 1
        nbr_compoenents_min = 1
        for i in tqdm.tqdm(range(1, 192)):
            if type == 'data_splited':
                self.train_model_pca('data_splited', num_comp=i)
                validation_loss = self.calculate_validation_loss_pca_data()
            else:
                self.train_model_pca('all_data', num_comp=i, save=False) #save=False so that we don't
                validation_loss = self.calculate_training_loss_pca_data() # save the model in each iteration

            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i

        print("nbr_comp_min: ",nbr_compoenents_min)
        self.train_model_pca('all_data', num_comp=nbr_compoenents_min)

    def train_model_pca(self, type='all_data', num_comp=167, save=True):
        self.data_man.load_pca_data(num_components=num_comp)
        if type == 'data_splited':
            data_pca_transformed_splited_train = self.data_man.get_data_pca_transformed_splited_train()
            labels_train = self.data_man.get_labels_splited_train()
            self.clf_pca_model_splited = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_pca_model_splited.fit(data_pca_transformed_splited_train, labels_train)
        else:
            data_pca_transformed = self.data_man.get_data_pca_transformed()
            labels = self.data_man.get_labels()
            self.clf_pca_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_pca_model.fit(data_pca_transformed, labels)
            if save == True: self.save_model_pca(numm_comp=num_comp)

    def calculate_training_loss_pca_data(self):
        labels = self.data_man.get_labels()
        data_pca_transformed = self.data_man.get_data_pca_transformed()
        probas_pca_data = self.clf_pca_model.predict_proba(data_pca_transformed)
        training_loss_pca_data = log_loss(y_true=labels, y_pred=probas_pca_data,
                                               labels=self.clf_pca_model.classes_)
        return training_loss_pca_data

    def calculate_validation_loss_pca_data(self):
        labels_test = self.data_man.get_labels_splited_test()
        data_pca_transformed_splited_test = self.data_man.get_data_pca_transformed_splited_test()
        probas_pca_data_test = self.clf_pca_model_splited.predict_proba(data_pca_transformed_splited_test)
        self.training_loss_pca_data_test = log_loss(y_true=labels_test, y_pred=probas_pca_data_test,
                                                    labels=self.clf_pca_model_splited.classes_)
        return self.training_loss_pca_data_test

    def save_model_pca(self,numm_comp=167):
        joblib.dump(self.clf_pca_model, '../models/lregr_pca_model_'+str(numm_comp)+'.joblib')

    def load_model_pca(self, num_comp=167):
        self.data_man.load_pca_data(num_components=num_comp)
        self.clf_pca_model = joblib.load('../models/lregr_pca_model_'+str(num_comp)+'.joblib')

    def submit_test_results_pca(self, nbr_comp_pca_model_trained_with=167):
        data_unlabeled_pca_transformed = self.data_man.get_unlabeled_pca_transformed()
        probas_unlabeled_pca = self.clf_pca_model.predict_proba(data_unlabeled_pca_transformed)
        header = self.clf_pca_model.classes_
        df = pd.DataFrame(probas_unlabeled_pca, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_pca_nbcomp_'+str(nbr_comp_pca_model_trained_with)+'.csv', index=None)
