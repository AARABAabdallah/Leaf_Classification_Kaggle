from sklearn.linear_model import LogisticRegression
import data.data_manipulation as dm
from sklearn.metrics import log_loss
import joblib
import pandas as pd
import numpy as np
import tqdm as tqdm
import keras
import keras.models
import keras.layers
import tensorflow as tf
import enlighten

class RN_model:
    def __init__(self):
        self.model_all_data = None
        self.model_images_model = None
        self.model_images_features_model = None
        self.model_pca_model_splited = None
        self.nbr_comp_pca_model_trained_with = None
        self.data_man = dm.DataManipulation()
        self.data_man.load_data()

    ############################# Features Trained model #############################
    def train_model_features(self):
        data, labels = self.data_man.get_data()
        # Convert labels to categorical one-hot encoding
        char_to_int = dict((c, i) for i,c in enumerate(set(list(labels))))
        labels_int_encoded = [char_to_int[char] for char in labels]
        one_hot_labels = tf.keras.utils.to_categorical(np.array(labels_int_encoded),num_classes = 99)

        self.model_all_data = tf.keras.models.Sequential()
        n_cols = data.shape[1]

        #add model layers
        self.model_all_data.add(tf.keras.layers.Dense(150, activation='selu', input_shape=(n_cols,)))
        # self.model_all_data.add(tf.keras.layers.Dense(125, activation='selu'))
        self.model_all_data.add(tf.keras.layers.Dense(99, activation='softmax'))
        self.model_all_data.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

        self.model_all_data_history = self.model_all_data.fit(data, one_hot_labels,
                                                              epochs=50, batch_size=50,validation_split=0.2)
        self.save_model_features()

    def get_training_loss(self):
        return self.model_all_data_history['loss']

    def save_model_features(self):
        self.model_all_data.save('../models/RN_all_data_model.h5')

    def load_model_features(self):
        self.model_all_data = tf.keras.models.load_model('../models/RN_all_data_model.h5')

    def submit_test_results_features(self):
        # Here we write the submission.csv file according to this model to be submitted in kaggle
        data_unlabeled = self.data_man.get_unlabeled_data()
        predictions_unlabeled = self.model_all_data.predict(data_unlabeled)
        data, labels = self.data_man.get_data()
        header = set(list(labels))
        df = pd.DataFrame(predictions_unlabeled, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/RN_test_results.csv', index=None)
 
    ############################# Images trained model CNN #############################
    def train_CNN_model_images(self):
        images_train = self.data_man.get_images_2D_data_train()
        labels = self.data_man.get_labels()
        # Convert labels to categorical one-hot encoding
        char_to_int = dict((c, i) for i, c in enumerate(set(list(labels))))
        labels_int_encoded = [char_to_int[char] for char in labels]
        one_hot_labels = tf.keras.utils.to_categorical(
            np.array(labels_int_encoded), num_classes=99)

        self.model_cnn_images_all_data = tf.keras.models.Sequential()
        n_cols = images_train.shape[1]
        
        self.model_cnn_images_all_data.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=(80, 80,1)))


        self.model_cnn_images_all_data.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model_cnn_images_all_data.add(tf.keras.layers.BatchNormalization())
        self.model_cnn_images_all_data.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model_cnn_images_all_data.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model_cnn_images_all_data.add(tf.keras.layers.BatchNormalization())
        self.model_cnn_images_all_data.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model_cnn_images_all_data.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model_cnn_images_all_data.add(tf.keras.layers.BatchNormalization())
        self.model_cnn_images_all_data.add(tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation='relu'))
        self.model_cnn_images_all_data.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model_cnn_images_all_data.add(tf.keras.layers.BatchNormalization())
        self.model_cnn_images_all_data.add(tf.keras.layers.Dropout(0.2))
        self.model_cnn_images_all_data.add(tf.keras.layers.Flatten())
        self.model_cnn_images_all_data.add(tf.keras.layers.Dense(120, activation='relu'))
        self.model_cnn_images_all_data.add(tf.keras.layers.Dropout(0.4))
        self.model_cnn_images_all_data.add(tf.keras.layers.Dense(99, activation='softmax'))

        self.model_cnn_images_all_data.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_cnn_images_all_data.fit(
            images_train.reshape((990, 80, 80, 1)), one_hot_labels, batch_size=80, epochs=60, verbose=1, validation_split = 0.2)
        self.save_CNN_model_images()


    def save_CNN_model_images(self):
        self.model_cnn_images_all_data.save(
            '../models/CNN_images_data_model.h5')

    def load_CNN_model_images(self):
        self.model_cnn_images_all_data = tf.keras.models.load_model(
            '../models/CNN_images_data_model.h5')

    def submit_test_results_CNN_images(self):
        labels = self.data_man.get_labels()
        images_unlabeled = self.data_man.get_images_2D_data_unlabeled()

        probas_unlabeled_images = self.model_cnn_images_all_data.predict(
            images_unlabeled.reshape((594, 80, 80, 1)).astype(float))
        header = set(list(labels))
        df = pd.DataFrame(probas_unlabeled_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/CNN_test_results_images.csv', index=None)

    ############################# Images trained model #############################

    def train_model_images(self):
        images_train = self.data_man.get_images_data_train()
        labels = self.data_man.get_labels()
        # Convert labels to categorical one-hot encoding
        char_to_int = dict((c, i) for i, c in enumerate(set(list(labels))))
        labels_int_encoded = [char_to_int[char] for char in labels]
        one_hot_labels = tf.keras.utils.to_categorical(
            np.array(labels_int_encoded), num_classes=99)

        self.model_images_all_data = tf.keras.models.Sequential()
        n_cols = images_train.shape[1]
        
        #add model layers
        self.model_images_all_data.add(tf.keras.layers.Dense(
            150, activation='relu', input_shape=(n_cols,)))
        self.model_images_all_data.add(tf.keras.layers.Dense(200, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(400, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(300, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(250, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(250, activation='relu'))
        self.model_images_all_data.add(
                tf.keras.layers.Dense(120, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(120, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(120, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(120, activation='relu'))
        self.model_images_all_data.add(
            tf.keras.layers.Dense(99, activation='softmax'))
        self.model_images_all_data.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_images_all_data.fit(
            images_train, one_hot_labels, batch_size=80, epochs=60, verbose=1, validation_split=0.2)
        self.save_model_images()

    def save_model_images(self):
        self.model_images_all_data.save(
            '../models/RN_images_data_model.h5')

    def load_model_images(self):
        self.model_images_all_data = tf.keras.models.load_model(
            '../models/RN_images_data_model.h5')

    def submit_test_results_images(self):
        labels = self.data_man.get_labels()
        images_unlabeled = self.data_man.get_images_data_unlabeled()

        probas_unlabeled_images = self.model_images_all_data.predict(
            images_unlabeled.astype(float))
        header = set(list(labels))
        df = pd.DataFrame(probas_unlabeled_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(
            r'../data_sets/submissions/RN_test_results_images.csv', index=None)


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

        # Convert labels to categorical one-hot encoding
        char_to_int = dict((c, i) for i, c in enumerate(set(list(labels))))
        labels_int_encoded = [char_to_int[char] for char in labels]
        one_hot_labels = tf.keras.utils.to_categorical(
            np.array(labels_int_encoded), num_classes=len(set(list(labels))))

        self.model_images_features_all_data = tf.keras.models.Sequential()
        n_cols = features_images_train.shape[1]

        #add model layers
        self.model_images_features_all_data.add(tf.keras.layers.Dense(150, activation='selu', input_shape=(n_cols,)))
        self.model_images_features_all_data.add(tf.keras.layers.Dense(125, activation='selu'))
        self.model_images_features_all_data.add(tf.keras.layers.Dense(125, activation='selu'))
        self.model_images_features_all_data.add(tf.keras.layers.Dense(125, activation='selu'))
        self.model_images_features_all_data.add(tf.keras.layers.Dense(125, activation='selu'))
        self.model_images_features_all_data.add(tf.keras.layers.Dense(
            len(set(list(labels))), activation='softmax'))
        self.model_images_features_all_data.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_images_features_all_data.fit(
            features_images_train, one_hot_labels, batch_size=200, epochs=200, validation_split=0.1)
        self.save_model_features_images()

    def save_model_features_images(self):
        self.model_images_features_all_data.save( '../models/RN_features_images_data_model.h5')

    def load_model_features_images(self):
        self.model_images_features_all_data = tf.keras.models.load_model(
            '../models/RN_features_images_data_model.h5')

    def submit_test_results_images_features(self):
        features_images_unlabeled = self.concatenate_features_images_test()
        features_images_train, labels = self.concatenate_features_images_train()
        probas_unlabeled_features_images = self.model_images_features_all_data.predict_proba(
            features_images_unlabeled)
        header = set(list(labels))
        df = pd.DataFrame(probas_unlabeled_features_images, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/RN_test_results_images_features.csv', index=None)



    ############################# model based on PCA transformed data #############################
    def train_model_pca_cross_validation(self, type='all_data'):
        # the validation_loss is calculated according to the type:
        # if type == 'data_splited' then the training and the validassion loss will be done according to the splited data
        # else(type = 'all_data' then the training and the validassion loss will be done according to all the data
        err_val_min = 1
        nbr_compoenents_min = 1
        pbar = enlighten.Counter(total=(len(list(range(1, 192)))),desc='Basic', unit='ticks')
        for i in range(1, 192):
            if type == 'data_splited':
                self.train_model_pca('data_splited', num_comp=i)
                validation_loss = self.calculate_validation_loss_pca_data()
            else:
                # save=False so that we don't save the model in each iteration
                self.train_model_pca('all_data', num_comp=i, save=False, verbose = 0)
                validation_loss = self.model_pca_model_history.history['val_loss'][-1]
                print(validation_loss)
            if validation_loss < err_val_min:
                err_val_min = validation_loss
                nbr_compoenents_min = i
            pbar.update()  # mise à jour de l'avancement des boucles

        print("nbr_comp_min: ",nbr_compoenents_min)
        self.train_model_pca('all_data', num_comp=nbr_compoenents_min)
        return nbr_compoenents_min

    def train_model_pca(self, type='all_data', num_comp=86, save=True,verbose = 1):
        # type is a parameter that specifies wether we train the model on all the training data,
            # or only on the splited data training, will be needed in the cross validation function
        # num_comp: is a parameter that specifies the number the components to hold after the data transformation
        # save: is a parameter that specifies wether to save the model or not (only when type!='all_data')
        self.data_man.load_pca_data(num_components=num_comp)
        if type == 'data_splited':
            data_pca_transformed_splited_train = self.data_man.get_data_pca_transformed_splited_train()
            labels_train = self.data_man.get_labels_splited_train()

            # Convert labels to categorical one-hot encoding
            char_to_int = dict((c, i) for i, c in enumerate(set(list(labels_train))))
            labels_int_encoded = [char_to_int[char] for char in labels_train]
            one_hot_labels = tf.keras.utils.to_categorical(
                np.array(labels_int_encoded), num_classes=len(set(list(labels_train))))

            self.model_pca_model = tf.keras.models.Sequential()
            n_cols = data_pca_transformed_splited_train.shape[1]

            #add model layers
            self.model_pca_model.add(tf.keras.layers.Dense(150, activation='selu', input_shape=(n_cols,)))
            # self.model_pca_model.add(tf.keras.layers.Dense(125, activation='selu'))
            self.model_pca_model.add(tf.keras.layers.Dense(len(set(list(labels_train))), activation='softmax'))
            self.model_pca_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

            self.model_pca_model_history = self.model_pca_model.fit(data_pca_transformed_splited_train, one_hot_labels,
                                                                  epochs=50, batch_size=50, validation_split=0.2)

        else:
            data_pca_transformed = self.data_man.get_data_pca_transformed()
            labels_train = self.data_man.get_labels()
            
            # Convert labels to categorical one-hot encoding
            char_to_int = dict((c, i) for i, c in enumerate(set(list(labels_train))))
            labels_int_encoded = [char_to_int[char] for char in labels_train]
            one_hot_labels = tf.keras.utils.to_categorical(
                np.array(labels_int_encoded), num_classes=len(set(list(labels_train))))

            self.model_pca_model = tf.keras.models.Sequential()
            n_cols = data_pca_transformed.shape[1]

            #add model layers
            self.model_pca_model.add(tf.keras.layers.Dense(150, activation='relu', input_shape=(n_cols,)))
            # self.model_pca_model.add(tf.keras.layers.Dense(125, activation='relu'))
            self.model_pca_model.add(tf.keras.layers.Dense(len(set(list(labels_train))), activation='softmax'))
            self.model_pca_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

            self.model_pca_model_history = self.model_pca_model.fit(data_pca_transformed, one_hot_labels,
                                                                  epochs=50, batch_size=50, validation_split=0.2,verbose=verbose)

            if save == True:
                self.save_model_pca(numm_comp=num_comp)
                self.nbr_comp_pca_model_trained_with = num_comp


    def save_model_pca(self,numm_comp=86):
        self.model_pca_model.save('../models/RN_pca_model_'+str(numm_comp)+'.h5')

    def load_model_pca(self, num_comp=86):
        self.data_man.load_pca_data(num_components=num_comp)
        self.nbr_comp_pca_model_trained_with = num_comp
        self.model_pca_model = tf.keras.models.load_model('../models/RN_pca_model_'+str(num_comp)+'.h5')

    def submit_test_results_pca(self):
        data_unlabeled_pca_transformed = self.data_man.get_unlabeled_pca_transformed()
        # labels = self.data_man.get_labels()
        data, labels = self.data_man.get_data()
        probas_unlabeled_pca = self.model_pca_model.predict(data_unlabeled_pca_transformed)
        header = set(list(labels))
        print(len(header))
        df = pd.DataFrame(probas_unlabeled_pca, columns=header)
        test_ids = self.data_man.get_test_data_ids()
        df.insert(loc=0, column='id', value=test_ids)
        df.to_csv(r'../data_sets/submissions/RN_test_results_pca_nbcomp_'+str(self.nbr_comp_pca_model_trained_with)+'.csv', index=None) 
