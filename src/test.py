import models.logistical_regression_model as lr
import data.data_manipulation as dm
import matplotlib.pyplot as plt

def main():
    data_man = dm.DataManipulation()
    log_reg = lr.LogisticalRegressionModel()
    log_reg.load_data(type='all_data')
    log_reg.load_data(type='data_splited')
    log_reg.load_unlabeled_data()

    #log_reg.train_model(type='all_data')
    #log_reg.train_model(type='data_splited')

    log_reg.load_all_data_model()
    log_reg.load_splited_data_model()

    training_all_data_loss = log_reg.calculate_training_loss(type='all_data')
    training_splited_data_loss = log_reg.calculate_training_loss(type='data_splited')
    test_splited_data_loss = log_reg.calculate_test_loss_splited_data()

    print(training_all_data_loss)
    print()
    print(training_splited_data_loss)
    print()
    print(test_splited_data_loss)
    #print(test_loss)
    #log_reg.save_all_data_model()
    # Test the leaf_image function
    #leaf_id = 343
    #leaf_img = data_man.leaf_image(leaf_id, target_length=160)
    #plt.imshow(leaf_img, cmap='gray')
    #plt.title('Leaf # ' + str(leaf_id))
    #plt.axis('off')
    #plt.show()

if __name__ == "__main__":
    main()
