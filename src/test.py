import data.data_manipulation as dm
import models.logistical_regression_model as lr
from sklearn.metrics import log_loss
import numpy as np

def main():
    log_reg = lr.LogisticalRegressionModel()
    #data_man = dm.DataManipulation()
    #x_train, y_train = data_man.load_data_train()
    #x_test = data_man.load_data_test()
    #clf = log_reg.train_model(x_train, y_train)
    #clf = log_reg.train_model()
    log_reg.train_model()

    training_loss = log_reg.calculate_loss()
    #probs = clf.predict_proba(x_train)
    #training_loss = log_loss(y_true=y_train, y_pred=probs, labels=clf.classes_)
    print(training_loss)
    #print(clf.classes_)
    #y = clf.predict(x_test[:10])
    #print()
    #print(y)

if __name__ == "__main__":
    main()
