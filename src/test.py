import models.logistical_regression_model as lr
import data.data_manipulation as dm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
#from PIL import Image
def main():
    data_man = dm.DataManipulation()
    log_reg = lr.LogisticalRegressionModel()
    log_reg.load_data(type='all_data')
    log_reg.load_data(type='data_splited')
    log_reg.load_unlabeled_data()

    log_reg.load_images_data_train()
    log_reg.load_images_data_unlabeled()

    log_reg.load_features_images_data_train()
    log_reg.load_features_images_data_unlabeled()
    #log_reg.train_model(type='all_data')
    #log_reg.train_model(type='data_splited')

    log_reg.load_all_data_model()
    log_reg.load_splited_data_model()

    training_all_data_loss = log_reg.calculate_training_loss(type='all_data')
    #training_splited_data_loss = log_reg.calculate_training_loss(type='data_splited')
    #test_splited_data_loss = log_reg.calculate_test_loss_splited_data()

    #log_reg.train_model_images()
    log_reg.load_images_data_model()
    #log_reg.save_images_data_model()
    training_data_images_loss = log_reg.calculate_training_loss_images()

    #log_reg.train_model_images_features()
    log_reg.load_features_images_data_model()

    training_data_images_features_loss = log_reg.calculate_training_loss_features_images()

    print(training_all_data_loss)
    print()
    print(training_data_images_loss)
    print()
    print(training_data_images_features_loss)
    #predictions = log_reg.predict_images_data(10)
    #print(predictions)
    #log_reg.submit_test_results()
    #log_reg.submit_test_results_images()
    log_reg.submit_test_results_images_features()
    #print(training_splited_data_loss)
    #print()
    #print(test_splited_data_loss)
    #print(test_loss)
    #log_reg.save_all_data_model()
    # Test the leaf_image function

    #leaf_id = 341
    #leaf_img = data_man.leaf_image(leaf_id, target_length=60)
    #print(np.shape(leaf_img))
    #df = pd.DataFrame(leaf_img)
    #images_train = data_man.load_images_data_train()
    #print(images_train)
    #print(np.unique(leaf_img))
    #print()
    #print(np.shape(leaf_img))
    #print()
    #plt.imshow(leaf_img, cmap='gray')
    #plt.title('Leaf # ' + str(leaf_id))
    #plt.axis('off')
    #plt.show()

if __name__ == "__main__":
    main()
