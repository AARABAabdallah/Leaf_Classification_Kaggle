import pandas as pd
import numpy as np


class DataManipulation:

    def __init__(self):
        self.NUMBER_OF_SPECIES = 99

    def load_data_train(self):

        raw_data = pd.read_csv("../data_sets/raw/train.csv")
        raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))] # Adding Bias !
        raw_data["ones"] = ones
        labels = raw_data["species"]
        labels = np.array(labels)
        raw_data = raw_data.drop("species", axis=1)
        raw_data = np.array(raw_data)
        self.data = raw_data
        self.labels = labels
        return raw_data, labels

    def load_data_test(self):
        raw_data = pd.read_csv("../data_sets/raw/test.csv")
        raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))]  # Adding Bias
        raw_data["ones"] = ones
        raw_data = np.array(raw_data)
        self.data_unlabeled = raw_data
        return raw_data

    def split_data(self):
        data_train,label_train = [],[]
        data_test,label_test = [],[]
        labels_uniq = np.unique(self.labels)
        dict_labels = {k:0 for k in list(labels_uniq)}

        for i in range(len(self.data)):
            if dict_labels[self.labels[i]]<8:
                data_train.append(self.data[i])
                label_train.append(self.labels[i])
            else:
                data_test.append(self.data[i])
                label_test.append(self.labels[i])
            dict_labels[self.labels[i]]+=1

        self.data_train,self.label_train = np.array(data_train),np.array(label_train)
        self.data_test,self.label_test = np.array(data_test),np.array(label_test)