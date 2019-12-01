import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn import preprocessing


class DataManipulation:

    def __init__(self):
        self.NUMBER_OF_SPECIES = 99
        self.data = None                    #Contains all the training data
        self.labels = None                  #Contains the labels of all the training data
        self.data_train = None              #Contains 80% of the trainging data
        self.data_test = None               #Contains 20% of the trainging data
        self.data_unlabeled = None          #Contains the testing data

        self.labels_train = None            #Contains labels that correspods to self.data_train
        self.labels_test = None             #Contains labels that correspods to self.data_test

        self.ids_all_data_train = None      #Contains a vector of ids that corresponds to self.data
        self.ids_splited_data_train = None  #Contains a vector of ids that corresponds to self.data_train
        self.ids_splited_data_test = None   #Contains a vector of ids that corresponds to self.data_test
        self.ids_data_test = None           #Contains a vector of ids that corresponds to self.data_unlabeled

        self.pca = None                     #Contains an instance of PCA class

        self.data_pca_transformed = None                #Contains self.data after pca transformation
        self.data_unlabeled_pca_transformed = None      #Contains self.data_unlabeled after pca transformation
        self.data_pca_transformed_splited_train = None  # Contains self.data_train(80% of data) after pca transformation
        self.data_pca_transformed_splited_test = None   # Contains self.data_test(20% of data) after pca transformation

        self.load_data()
        self.load_unlabeled_data()
        #self.data_normalization_centring()

    def load_data(self):
        # Here we prepare for later use the attributs: self.data, self.labels, self.ids_all_data_train,
        # self.data_train, self.labels_train, self.data_test, self.labels_test
        # self.ids_splited_data_train, self.ids_splited_data_test
        raw_data = pd.read_csv("../data_sets/raw/train.csv")
        labels = raw_data["species"]
        labels = np.array(labels)
        self.ids_all_data_train = raw_data["id"]
        self.ids_all_data_train = np.array(self.ids_all_data_train)
        raw_data = raw_data.drop("species", axis=1)
        raw_data = raw_data.drop("id", axis=1)
        raw_data = np.array(raw_data)

        #raw_data -= np.mean(self.data, axis=0) # data centring
        #raw_data /= np.std(self.data, axis=0)  # data normalization
        raw_data = preprocessing.scale(raw_data)

        self.data = raw_data
        self.labels = labels

        self.split_data()

    def load_unlabeled_data(self):
        raw_data = pd.read_csv("../data_sets/raw/test.csv")
        self.ids_data_test = raw_data["id"]
        raw_data = raw_data.drop("id", axis=1)
        raw_data = np.array(raw_data)

        #raw_data -= np.mean(self.data, axis=0)  # data centring
        #raw_data /= np.std(self.data, axis=0)  # data normalization
        raw_data = preprocessing.scale(raw_data)

        self.data_unlabeled = raw_data

    def split_data(self):
        # Prepare the data for the cross validation by spleiting the training data on the sub training data(80% of data)
        # and the validation data(20% of data)
        data_train, labels_train = [], []
        data_test, labels_test = [], []
        self.ids_splited_data_train, self.ids_splited_data_test = [], []
        labels_uniq = np.unique(self.labels)
        dict_labels = {k: 0 for k in list(labels_uniq)}

        for i in range(len(self.data)):
            if dict_labels[self.labels[i]] < 8:
                data_train.append(self.data[i])
                labels_train.append(self.labels[i])
                self.ids_splited_data_train.append(self.ids_all_data_train[i])
            else:
                data_test.append(self.data[i])
                labels_test.append(self.labels[i])
                self.ids_splited_data_test.append(self.ids_all_data_train[i])
            dict_labels[self.labels[i]] += 1

        self.data_train, self.labels_train = np.array(data_train), np.array(labels_train)
        self.data_test, self.labels_test = np.array(data_test), np.array(labels_test)
        self.ids_splited_data_train, self.ids_splited_data_test = np.array(self.ids_splited_data_train), np.array(
            self.ids_splited_data_test)

    def get_data(self):
        return self.data, self.labels

    def get_training_data(self):
        return self.data_train, self.labels_train

    def get_test_data(self):
        return self.data_test, self.labels_test

    def get_unlabeled_data(self):
        return self.data_unlabeled

    def get_test_data_ids(self):
        return self.ids_data_test

    def get_images_data_train(self):
        images_train = []
        for i in range(len(self.data)):
            img = self.leaf_image(self.ids_all_data_train[i])
            img = np.array(img)
            img = img.flatten()
            images_train.append(img)
        return np.array(images_train)
    
    def get_images_2D_data_train(self):
        images_train = []
        for i in range(len(self.data)):
            img = self.leaf_image(self.ids_all_data_train[i])
            img = np.array(img)
            images_train.append(img)
        return np.array(images_train)

    def get_images_data_unlabeled(self):
        images_test = []
        for i in range(len(self.data_unlabeled)):
            img = self.leaf_image(self.ids_data_test[i])
            img = np.array(img)
            img = img.flatten()
            images_test.append(img)
        return np.array(images_test)
    
    def get_images_2D_data_unlabeled(self):
        images_test = []
        for i in range(len(self.data_unlabeled)):
            img = self.leaf_image(self.ids_data_test[i])
            img = np.array(img)
            images_test.append(img)
        return np.array(images_test)

    def leaf_image(self, image_id, target_length=80):
        """
        `image_id` should be the index of the image in the images/ folder
        Reture the image of a given id(1~1584) with the target size (target_length x target_length)
        """
        image_name = str(image_id) + '.jpg'
        leaf_img = plt.imread('../data_sets/raw/images/' + image_name)  # Reading in the image
        leaf_img_width = leaf_img.shape[1]
        leaf_img_height = leaf_img.shape[0]
        # target_length = 160
        img_target = np.zeros((target_length, target_length), np.uint8)
        if leaf_img_width >= leaf_img_height:
            scale_img_width = target_length
            scale_img_height = int((float(scale_img_width) / leaf_img_width) * leaf_img_height)
            img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation=cv2.INTER_AREA)
            copy_location = int((target_length - scale_img_height) / 2)
            img_target[copy_location:copy_location + scale_img_height, :] = img_scaled
        else:
            # leaf_img_width < leaf_img_height:
            scale_img_height = target_length
            scale_img_width = int((float(scale_img_height) / leaf_img_height) * leaf_img_width)
            img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation=cv2.INTER_AREA)
            copy_location = int((target_length - scale_img_width) / 2)
            img_target[:, copy_location:copy_location + scale_img_width] = img_scaled

        return img_target

    def load_pca_data(self, num_components=167):
        # Here we trasform the data accordingly to the num_components parameter that specifies how many columns
        # we want after the transformation
        self.pca = PCA(n_components=num_components, svd_solver='full')
        self.pca.fit(self.data)#raw_data)
        self.data_pca_transformed = self.pca.transform(self.data)
        self.data_unlabeled_pca_transformed = self.pca.transform(self.data_unlabeled)
        self.data_pca_transformed_splited_train = self.pca.transform(self.data_train)#[:, :192])  # to exclude the ones added
        self.data_pca_transformed_splited_test = self.pca.transform(self.data_test)#[:, :192])

    def get_data_pca_transformed(self):
        return self.data_pca_transformed

    def get_unlabeled_pca_transformed(self):
        return self.data_unlabeled_pca_transformed

    def get_data_pca_transformed_splited_train(self):
        return self.data_pca_transformed_splited_train

    def get_data_pca_transformed_splited_test(self):
        return self.data_pca_transformed_splited_test

    def get_labels(self):
        return self.labels

    def get_labels_splited_train(self):
        return self.labels_train

    def get_labels_splited_test(self):
        return self.labels_test
