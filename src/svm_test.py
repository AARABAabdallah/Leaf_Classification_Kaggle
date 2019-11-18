import models.logistical_regression_model as lr
import models.svm_model as svc

def main():
    log_reg = lr.LogisticalRegressionModel()
    #log_reg.train_model(type='all_data')
    svm_model = svc.SvmModel()
    svm_model.cross_validation(kernel='poly')
    #svm_model.train_model(kernel='poly',gamma='auto')
    #svm_model.calcul_test_accuracy()
    #print(svm_model.test_accuracy)
    #training_loss = log_reg.calculate_training_loss(type= 'data_splited')
    #test_loss = log_reg.calculate_training_loss('data_splited')
    #print(training_loss)

if __name__ == "__main__":
    main()
