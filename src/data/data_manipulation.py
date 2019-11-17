import pandas as pd
import numpy as np

class DataManipulation:

    def load_data_train(self):

        raw_data = pd.read_csv("../data_sets/raw/train.csv")
        raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))] # Adding Bias !
        raw_data["ones"] = ones
        labels = raw_data["species"]
        labels = np.array(labels)
        raw_data = raw_data.drop("species", axis=1)
        raw_data = np.array(raw_data)
        return raw_data, labels

    def load_data_test(self):
        raw_data = pd.read_csv("../data_sets/raw/test.csv")
        raw_data = raw_data.drop("id", axis=1)
        ones = [1 for i in range(len(raw_data))]  # Adding Bias
        raw_data["ones"] = ones
        raw_data = np.array(raw_data)
        return raw_data
