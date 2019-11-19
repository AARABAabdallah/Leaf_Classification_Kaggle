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
        self.clf_splited_data = None
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

    def train_model(self, type='all_data'):
        if type == 'all_data':
            # self.load_data('all_data')
            self.clf_all_data = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_all_data.fit(self.data, self.labels)
            self.save_all_data_model()
        else:
            # self.load_data('data_splited')
            self.clf_splited_data = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_splited_data.fit(self.data_train, self.labels_train)
            self.save_splited_data_model()

    def load_data(self, type='data_splited'):
        self.data_man.load_data()
        self.data_man.load_unlabeled_data()
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

    def predict_proba_unlabeled(self):
        self.probas_unlabeled = self.clf_all_data.predict_proba(self.data_unlabeled)

    def calculate_training_loss(self, type='data_splited'):
        if type == 'all_data':
            self.predict_proba_train('all_data')
            self.training_loss_all_data = log_loss(y_true=self.labels, y_pred=self.probas_all_train_data,
                                                   labels=self.clf_all_data.classes_)
            return self.training_loss_all_data
        else:
            self.predict_proba_train('data_splited')
            self.training_loss_splited_data = log_loss(y_true=self.labels_train, y_pred=self.probas_train_data_splited,
                                                       labels=self.clf_splited_data.classes_)
            return self.training_loss_splited_data

    def calculate_test_loss_splited_data(self):
        self.data_test, self.labels_test = self.data_man.get_test_data()
        self.predict_proba_test_splited_data()
        self.test_loss_data_splited = log_loss(y_true=self.labels_test, y_pred=self.probas_test_data_splited,
                                               labels=self.clf_splited_data.classes_)
        return self.test_loss_data_splited

    def load_unlabeled_data(self):
        self.data_unlabeled = self.data_man.get_unlabeled_data()

    def save_all_data_model(self):
        joblib.dump(self.clf_all_data, '../models/lregr_all_data_model.joblib')

    def save_splited_data_model(self):
        joblib.dump(self.clf_splited_data, '../models/lregr_splited_data_model.joblib')

    def load_all_data_model(self):
        self.clf_all_data = joblib.load('../models/lregr_all_data_model.joblib')

    def load_splited_data_model(self):
        self.clf_splited_data = joblib.load('../models/lregr_splited_data_model.joblib')

    def load_images_data_train(self):
        self.images_train = self.data_man.load_images_data_train()

    def load_images_data_unlabeled(self):
        self.images_unlabeled = self.data_man.load_images_data_unlabeled()

    def load_features_images_data_train(self):
        self.features_images_train = np.concatenate((self.data, self.images_train), axis=1)

    def load_features_images_data_unlabeled(self):
        self.features_images_unlabeled = np.concatenate((self.data_unlabeled, self.images_unlabeled), axis=1)

    def submit_test_results(self):
        self.predict_proba_unlabeled()
        header = self.clf_all_data.classes_
        df = pd.DataFrame(self.probas_unlabeled, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results.csv', index=None)

    def submit_test_results_images(self):
        self.predict_probas_unlabeled_images()
        header = self.clf_images_model.classes_
        df = pd.DataFrame(self.probas_unlabeled_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_images.csv', index=None)

    def submit_test_results_images_features(self):
        self.predict_probas_unlabeled_features_images()
        header = self.clf_images_features_model.classes_
        df = pd.DataFrame(self.probas_unlabeled_features_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_images_features.csv', index=None)

    def train_model_images(self):
        # self.images_train = self.data_man.load_images_data_train()
        self.clf_images_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        self.clf_images_model.fit(self.images_train, self.labels)
        self.save_images_data_model()

    def predict_probas_train_images(self):
        self.probas_data_train_images = self.clf_images_model.predict_proba(self.images_train)

    def predict_probas_train_features_images(self):
        self.probas_data_train_features_images = self.clf_images_features_model.predict_proba(
            self.features_images_train)

    def predict_probas_unlabeled_images(self):
        self.probas_unlabeled_images = self.clf_images_model.predict_proba(self.images_unlabeled)

    def predict_probas_unlabeled_features_images(self):
        self.probas_unlabeled_features_images = self.clf_images_features_model.predict_proba(
            self.features_images_unlabeled)

    def calculate_training_loss_images(self):
        self.predict_probas_train_images()
        self.training_loss_data_train_images = log_loss(y_true=self.labels, y_pred=self.probas_data_train_images,
                                                        labels=self.clf_images_model.classes_)
        return self.training_loss_data_train_images

    def save_images_data_model(self):
        joblib.dump(self.clf_images_model, '../models/lregr_images_data_model.joblib')

    def load_images_data_model(self):
        self.clf_images_model = joblib.load('../models/lregr_images_data_model.joblib')

    def train_model_images_features(self):
        self.clf_images_features_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
        self.clf_images_features_model.fit(self.features_images_train, self.labels)
        self.save_features_images_data_model()

    def calculate_training_loss_features_images(self):
        self.predict_probas_train_features_images()
        self.training_loss_data_train_features_images = log_loss(y_true=self.labels,
                                                                 y_pred=self.probas_data_train_features_images,
                                                                 labels=self.clf_images_model.classes_)
        return self.training_loss_data_train_features_images

    def save_features_images_data_model(self):
        joblib.dump(self.clf_images_features_model, '../models/lregr_features_images_data_model.joblib')

    def load_features_images_data_model(self):
        self.clf_images_features_model = joblib.load('../models/lregr_features_images_data_model.joblib')

    #### model based on PCA transformed data
    def load_pca_data(self, num_comp=140):
        self.nbr_comp_pca_model_trained_with = num_comp
        self.data_pca_transformed, self.data_unlabeled_pca_transformed, self.data_pca_transformed_splited_train, self.data_pca_transformed_splited_test = self.data_man.load_pca_data(
            num_components=num_comp)

    def train_pca_model_cross_validation(self):
        err_val_min = 1
        nbr_compoenents_min = 1
        for i in tqdm.tqdm(range(1, 192)):
            self.load_pca_data(num_comp=i)
            self.train_pca_model('data_splited')
            validation_loss = self.calculate_training_loss_pca_data_test()
            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i
        print("nbr_comp_min: ",nbr_compoenents_min)
        self.load_pca_data(num_comp=nbr_compoenents_min)
        self.train_pca_model('all_data')

    def train_pca_model(self, type='all_data'):
        if type == 'data_splited':
            self.data_test, self.labels_test = self.data_man.get_test_data()
            self.clf_pca_model_splited = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_pca_model_splited.fit(self.data_pca_transformed_splited_train, self.labels_train)
        else:
            self.clf_pca_model = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial')
            self.clf_pca_model.fit(self.data_pca_transformed, self.labels)
            self.save_pca_model()

    def save_pca_model(self):
        joblib.dump(self.clf_pca_model, '../models/lregr_pca_model.joblib')

    def load_pca_model(self):
        self.clf_pca_model = joblib.load('../models/lregr_pca_model.joblib')

    def calculate_training_loss_pca_data(self):
        self.predict_probas_pca_data()
        self.training_loss_pca_data = log_loss(y_true=self.labels, y_pred=self.probas_pca_data,
                                               labels=self.clf_pca_model.classes_)
        return self.training_loss_pca_data

    def calculate_training_loss_pca_data_train(self):
        self.predict_probas_pca_data_train()
        self.training_loss_pca_data_train = log_loss(y_true=self.labels_train, y_pred=self.probas_pca_data_train,
                                                     labels=self.clf_pca_model_splited.classes_)
        return self.training_loss_pca_data_train

    def calculate_training_loss_pca_data_test(self):
        self.predict_probas_pca_data_test()
        self.training_loss_pca_data_test = log_loss(y_true=self.labels_test, y_pred=self.probas_pca_data_test,
                                                    labels=self.clf_pca_model_splited.classes_)
        return self.training_loss_pca_data_test

    def predict_probas_pca_data(self):
        self.probas_pca_data = self.clf_pca_model.predict_proba(self.data_pca_transformed)

    def predict_probas_pca_data_train(self):
        self.probas_pca_data_train = self.clf_pca_model_splited.predict_proba(self.data_pca_transformed_splited_train)

    def predict_probas_pca_data_test(self):
        self.probas_pca_data_test = self.clf_pca_model_splited.predict_proba(self.data_pca_transformed_splited_test)

    def predict_probas_unlabeled_pca(self):
        self.probas_unlabeled_pca = self.clf_pca_model.predict_proba(self.data_unlabeled_pca_transformed)

    def submit_test_results_pca(self):
        self.predict_probas_unlabeled_pca()
        header = self.clf_pca_model.classes_
        df = pd.DataFrame(self.probas_unlabeled_pca, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/lr_test_results_pca_nbcomp_'+str(self.nbr_comp_pca_model_trained_with)+'.csv', index=None)
