import data.data_manipulation as dm
import models.logistical_regression_model as lr
from sklearn.metrics import log_loss
import numpy as np

def main():
    log_reg = lr.LogisticalRegressionModel()
    log_reg.train_model()
    training_loss = log_reg.calculate_loss()
    print(training_loss)

if __name__ == "__main__":
    main()
