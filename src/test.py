import models.logistical_regression_model as lr

def main():
    log_reg = lr.LogisticalRegressionModel()
    log_reg.load_data(type='data_splited')
    #log_reg.train_model(type='data_splited')
    #log_reg.train_model(type='data_splited')
    #training_loss = log_reg.calculate_training_loss(type= 'all_data')
    log_reg.load_splited_data_model()
    training_loss = log_reg.calculate_training_loss(type='data_splited')
    test_loss = log_reg.calculate_test_loss_splited_data()
    print(training_loss)
    print()
    print(test_loss)
    #log_reg.save_splited_data_model()

if __name__ == "__main__":
    main()
