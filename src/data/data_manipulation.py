import pandas as pd
import numpy as np

class DataManipulation:

    def load_data_train(self):

        raw_data = pd.read_csv("../data_sets/raw/train.csv")
        raw_data = raw_data.drop("id", axis=1)
        """
        acer = raw_data['species'] == 'Acer_Opalus'#, 'Pterocarya_Stenoptera', 'Quercus_Hartwissiana']
        pter = raw_data['species'] == 'Pterocarya_Stenoptera'#, 'Pterocarya_Stenoptera', 'Quercus_Hartwissiana']
        quer = raw_data['species'] == 'Quercus_Hartwissiana'#, 'Pterocarya_Stenoptera', 'Quercus_Hartwissiana']
        data = raw_data[acer | pter | quer]
        #labels = self.generate_labels(data)
        #labels = np.array(labels)
        labels = data["species"]
        labels = np.array(labels)
        data = data.drop("species", axis=1)
        data = np.array(data)
        """
        labels = raw_data["species"]
        labels = np.array(labels)
        raw_data = raw_data.drop("species", axis=1)
        raw_data = np.array(raw_data)
        return raw_data, labels

    def load_data_test(self):
        raw_data = pd.read_csv("../data_sets/raw/test.csv")
        raw_data = raw_data.drop("id", axis=1)
        #labels = raw_data["species"]
        #labels = np.array(labels)
        #raw_data = raw_data.drop("species", axis=1)
        raw_data = np.array(raw_data)
        return raw_data
    """
    def generate_labels(self, data):
        labels = []

        for i in range(len(data)):
            t = []
            if data.iloc[i]['species'] == 'Acer_Opalus':
                t = [1, 0, 0]
            elif data.iloc[i]['species'] == 'Pterocarya_Stenoptera':
                t = [0, 1, 0]
            else:
                t = [0, 0, 1]
            labels.append(t)
        return labels
    """