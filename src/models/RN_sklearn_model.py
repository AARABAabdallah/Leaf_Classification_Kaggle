from sklearn.neural_network import MLPClassifier
import data.data_manipulation as dm
from sklearn.metrics import log_loss
import joblib
import pandas as pd
import numpy as np
import tqdm as tqdm
import enlighten

class RN_sklearn_model:
    def __init__(self):
        self.RN_skl_all_data = None
        self.RN_skl_images_model = None
        self.RN_skl_images_features_model = None
        self.RN_skl_pca_model_splited = None
        self.nbr_comp_pca_model_trained_with = None
        self.data_man = dm.DataManipulation()
        #self.data_man.load_data()

    ############################# Features Trained model #############################
    def features_cross_valid(self, layer1_range=range(180, 221, 10), layer2_range=range(110, 161, 10), layer3_range=range(100,201,10)):
        layer1_range = layer1_range
        layer2_range = [0]+list(layer2_range)
        layer3_range = [0]+list(layer3_range)
        best_layers_size=(9)
        self.train_model_features(save=False,layers_size=best_layers_size)
        min_loss = self.get_training_loss()

        pbar = enlighten.Counter(total=(len(layer2_range)*len(layer1_range)*len(layer3_range)),desc='Basic', unit='ticks')
        for nbr_lyr3 in layer3_range:
            for nbr_lyr2 in layer2_range:
                for nbr_lyr1 in layer1_range:
                    if nbr_lyr2 == 0 :
                        self.train_model_features(save=False, layers_size=(nbr_lyr1))
                        if self.get_training_loss() < min_loss:
                            min_loss = self.get_training_loss()
                            best_layers_size = (nbr_lyr1)
                            print('min_loss : ', min_loss,' -- layers_size : ',best_layers_size)
                    elif nbr_lyr3 == 0:
                        self.train_model_features(save=False, layers_size=(nbr_lyr1,nbr_lyr2))
                        if self.get_training_loss() < min_loss:
                            min_loss = self.get_training_loss()
                            best_layers_size = (nbr_lyr1,nbr_lyr2)
                            print('min_loss : ', min_loss,' -- layers_size : ',best_layers_size)
                    else :
                        self.train_model_features(save=False, layers_size=(nbr_lyr1,nbr_lyr2,nbr_lyr3))
                        if self.get_training_loss() < min_loss:
                            min_loss = self.get_training_loss()
                            best_layers_size = (nbr_lyr1,nbr_lyr2,nbr_lyr3)
                            print('min_loss : ', min_loss,' -- layers_size : ',best_layers_size)
                    pbar.update()  # mise à jour de l'avancement des boucles
        
        self.train_model_features(layers_size=best_layers_size)
        print("\n\nbest layer size : ",best_layers_size)
        print("min loss : ",min_loss)

        
    def train_model_features(self,save=True,layers_size=(220,160)):
        data, labels = self.data_man.get_data()
        self.RN_skl_all_data = MLPClassifier(hidden_layer_sizes=layers_size, activation='relu', solver='adam')
        self.RN_skl_all_data.fit(data, labels)
        if save : self.save_model_features()

    def get_training_loss(self):
        return self.RN_skl_all_data.loss_


    def save_model_features(self):
        joblib.dump(self.RN_skl_all_data, '../models/RN_skl_all_data.joblib')

    def load_model_features(self):
        self.RN_skl_all_data = joblib.load('../models/RN_skl_all_data.joblib')

    def submit_test_results_features(self):
        # Here we write the submission.csv file according to this model to be submitted in kaggle
        data_unlabeled = self.data_man.get_unlabeled_data()
        probas_unlabeled = self.RN_skl_all_data.predict_proba(data_unlabeled)
        header = self.RN_skl_all_data.classes_
        df = pd.DataFrame(probas_unlabeled, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/RN_skl_test_results.csv', index=None)

############################# Images trained model #############################
    def train_model_images(self):
        images_train = self.data_man.get_images_data_train()
        labels = self.data_man.get_labels()
        self.RN_skl_images_model = MLPClassifier(hidden_layer_sizes=(150,130,100,),activation='relu',solver='adam')
        self.RN_skl_images_model.fit(images_train, labels)
        self.save_model_images()

    def get_training_loss_images(self):
        return self.RN_skl_images_model.loss_

    def save_model_images(self):
        joblib.dump(self.RN_skl_images_model,
                    '../models/RN_skl_images_data_model.joblib')

    def load_model_images(self):
        self.RN_skl_images_model = joblib.load(
            '../models/RN_skl_images_data_model.joblib')

    def submit_test_results_images(self):
        images_unlabeled = self.data_man.get_images_data_unlabeled()
        probas_unlabeled_images = self.RN_skl_images_model.predict_proba(
            images_unlabeled)
        header = self.RN_skl_images_model.classes_
        df = pd.DataFrame(probas_unlabeled_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(
            r'../data_sets/submissions/RN_skl_test_results_images.csv', index=None)

    ############################# Features, Images concatenated trained model #############################

    def concatenate_features_images_train(self):
        # Here we concatenate features columns and images columns for the training data
        data, labels = self.data_man.get_data()
        images_train = self.data_man.get_images_data_train()
        features_images_train = np.concatenate((data, images_train), axis=1)
        return features_images_train, labels

    def concatenate_features_images_test(self):
        # Here we concatenate features columns and images columns for the testing data
        data_test = self.data_man.get_unlabeled_data()
        images_test = self.data_man.get_images_data_unlabeled()
        features_images_test = np.concatenate((data_test, images_test), axis=1)
        return features_images_test

    def train_model_images_features(self):
        features_images_train, labels = self.concatenate_features_images_train()
        self.RN_skl_images_features_model = MLPClassifier(hidden_layer_sizes=(220,160),activation='relu',solver='adam')
        self.RN_skl_images_features_model.fit(features_images_train, labels)
        self.save_model_features_images()

    def get_training_loss_features_images(self):
        return self.RN_skl_images_features_model.loss_

    def save_model_features_images(self):
        joblib.dump(self.RN_skl_images_features_model,
                    '../models/RN_skl_features_images_data_model.joblib')

    def load_model_features_images(self):
        self.RN_skl_images_features_model = joblib.load(
            '../models/RN_skl_features_images_data_model.joblib')

    def submit_test_results_images_features(self):
        features_images_unlabeled = self.concatenate_features_images_test()
        probas_unlabeled_features_images = self.RN_skl_images_features_model.predict_proba(
            features_images_unlabeled)
        header = self.RN_skl_images_features_model.classes_
        df = pd.DataFrame(probas_unlabeled_features_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(
            r'../data_sets/submissions/RN_skl_test_results_images_features.csv', index=None)

    ############################# model based on PCA transformed data #############################
    def train_model_pca_cross_validation(self, type='data_splited'):
        # the validation_loss is calculated according to the type:
        # if type == 'data_splited' then the training and the validassion loss will be done according to the splited data
        # else(type = 'all_data' then the training and the validassion loss will be done according to all the data
        err_val_min = 100
        nbr_compoenents_min = 1
        for i in tqdm.tqdm(range(1, 192)):
            if type == 'data_splited':
                self.train_model_pca('data_splited', num_comp=i)
                validation_loss = self.calculate_validation_loss_pca_data()
            else:
                # save=False so that we don't save the model in each iteration
                self.train_model_pca('all_data', num_comp=i, save=False)
                validation_loss = self.get_training_loss_pca_data()

            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i
                print('err_val_min : ', err_val_min, ' -- nbr_compoenents_min : ',nbr_compoenents_min)

        print("best nbr_comp_min: ", nbr_compoenents_min)
        self.train_model_pca('all_data', num_comp=nbr_compoenents_min)
        return nbr_compoenents_min

    def train_model_pca(self, type='all_data', num_comp=24, save=True):
        # type is a parameter that specifies wether we train the model on all the training data,
            # or only on the splited data training, will be needed in the cross validation function
        # num_comp: is a parameter that specifies the number the components to hold after the data transformation
        # save: is a parameter that specifies wether to save the model or not (only when type!='all_data')
        self.data_man.load_pca_data(num_components=num_comp)
        if type == 'data_splited':
            data_pca_transformed_splited_train = self.data_man.get_data_pca_transformed_splited_train()
            labels_train = self.data_man.get_labels_splited_train()
            self.RN_skl_pca_model_splited = MLPClassifier(hidden_layer_sizes=(220,160),activation='relu',solver='adam')
            self.RN_skl_pca_model_splited.fit(
                data_pca_transformed_splited_train, labels_train)
        else:
            data_pca_transformed = self.data_man.get_data_pca_transformed()
            labels = self.data_man.get_labels()
            self.RN_skl_pca_model = MLPClassifier(hidden_layer_sizes=(220,160),activation='relu',solver='adam')
            self.RN_skl_pca_model.fit(data_pca_transformed, labels)
            if save == True:
                self.save_model_pca(num_comp=num_comp)
                self.nbr_comp_pca_model_trained_with = num_comp

    def get_training_loss_pca_data(self):
        return self.RN_skl_pca_model.loss_

    def calculate_validation_loss_pca_data(self):
        labels_test = self.data_man.get_labels_splited_test()
        data_pca_transformed_splited_test = self.data_man.get_data_pca_transformed_splited_test()
        probas_pca_data_test = self.RN_skl_pca_model_splited.predict_proba(
            data_pca_transformed_splited_test)
        self.training_loss_pca_data_test = log_loss(y_true=labels_test, y_pred=probas_pca_data_test,
                                                    labels=self.RN_skl_pca_model_splited.classes_)
        return self.training_loss_pca_data_test

    def save_model_pca(self, num_comp=24):
        joblib.dump(self.RN_skl_pca_model,
                    '../models/RN_skl_pca_model_'+str(num_comp)+'.joblib')

    def load_model_pca(self, num_comp=24):
        self.data_man.load_pca_data(num_components=num_comp)
        self.nbr_comp_pca_model_trained_with = num_comp
        self.RN_skl_pca_model = joblib.load(
            '../models/RN_skl_pca_model_'+str(num_comp)+'.joblib')

    def submit_test_results_pca(self):
        data_unlabeled_pca_transformed = self.data_man.get_unlabeled_pca_transformed()
        probas_unlabeled_pca = self.RN_skl_pca_model.predict_proba(
            data_unlabeled_pca_transformed)
        header = self.RN_skl_pca_model.classes_
        df = pd.DataFrame(probas_unlabeled_pca, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/RN_skl_test_results_pca_nbcomp_' +
                  str(self.nbr_comp_pca_model_trained_with)+'.csv', index=None)
