import pandas as pd
import numpy as np


class DataManipulation:

    def __init__(self):
        self.NUMBER_OF_SPECIES = 99
        self.data = None
        self.labels = None
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None
        self.data_unlabeled = None
        self.ids_all_data_train = None
        self.ids_splited_data_train = None
        self.ids_splited_data_test = None
        self.ids_data_test = None


    def load_data(self):
        raw_data = pd.read_csv("../data_sets/raw/train.csv")
        #raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))] # Adding Bias !
        raw_data["ones"] = ones
        labels = raw_data["species"]
        labels = np.array(labels)
        self.ids_all_data_train = raw_data["id"]
        self.ids_all_data_train = np.array(self.ids_all_data_train)
        raw_data = raw_data.drop("species", axis=1)
        raw_data = raw_data.drop("id", axis=1)
        raw_data = np.array(raw_data)
        self.data = raw_data
        self.labels = labels
        self.split_data()

    def load_unlabeled_data(self):
        raw_data = pd.read_csv("../data_sets/raw/test.csv")
        self.ids_data_test = raw_data["id"]
        raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))]  # Adding Bias
        raw_data["ones"] = ones
        raw_data = np.array(raw_data)
        self.data_unlabeled = raw_data

    def split_data(self):
        data_train,labels_train = [],[]
        data_test,labels_test = [],[]
        self.ids_splited_data_train, self.ids_splited_data_test = [],[]
        labels_uniq = np.unique(self.labels)
        dict_labels = {k:0 for k in list(labels_uniq)}

        for i in range(len(self.data)):
            if dict_labels[self.labels[i]]<8:
                data_train.append(self.data[i])
                labels_train.append(self.labels[i])
                self.ids_splited_data_train.append(self.ids_all_data_train[i])
            else:
                data_test.append(self.data[i])
                labels_test.append(self.labels[i])
                self.ids_splited_data_test.append(self.ids_all_data_train[i])
            dict_labels[self.labels[i]]+=1

        self.data_train,self.labels_train = np.array(data_train),np.array(labels_train)
        self.data_test,self.labels_test = np.array(data_test),np.array(labels_test)
        self.ids_splited_data_train, self.ids_splited_data_test = np.array(self.ids_splited_data_train), np.array(self.ids_splited_data_test)

    def get_data(self):
        return self.data, self.labels

    def get_training_data(self):
        return self.data_train, self.labels_train

    def get_test_data(self):
        return self.data_test, self.labels_test

    def get_unlabeled_data(self):
        return self.data_unlabeled