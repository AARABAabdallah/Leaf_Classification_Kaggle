import data.data_manipulation as dm
import models.logistical_regression_model as lr
from sklearn.linear_model import LogisticRegression

def main():
    log_reg = lr.LogisticalRegressionModel()
    data_man = dm.DataManipulation()
    x_train, y_train = data_man.load_data_train()
    x_test = data_man.load_data_test()
    clf = log_reg.train_model(x_train, y_train)
    y = clf.predict(x_test[:10])
    print(y)

if __name__ == "__main__":
    main()
