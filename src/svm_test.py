import models.logistical_regression_model as lr
import models.svm_model as svc

def main():
    #log_reg = lr.LogisticalRegressionModel()
    #log_reg.train_model(type='all_data')
    svm_model = svc.SvmModel()
    #svm_model.train_model(kernel='rbf',gamma='auto')
    # svm_model.train_model_pca_cross_validation(kernel="poly",degree=1,coef0=0,gamma=0.1)
    # svm_model.load_model_pca(num_comp=9)
    svm_model.train_model_pca_cross_validation(kernel='sigmoid',gamma='auto',coef0=4)
    print(svm_model.calculate_training_loss_pca_data())
    # svm_model.submit_test_results_pca()
    # svm_model.cross_validation(kernel='sigmoid')
    # svm_model.submit_test_results_features()
    # print()
    #svm_model.calcul_test_accuracy()
    #print(svm_model.test_accuracy)
    #training_loss = log_reg.calculate_training_loss(type= 'data_splited')
    #test_loss = log_reg.calculate_training_loss('data_splited')
    #print(training_loss)

if __name__ == "__main__":
    main()
