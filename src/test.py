import tqdm as tqdm

import models.logistical_regression_model as lr
import data.data_manipulation as dm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd


# from PIL import Image

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
    # log_reg.train_model(type='all_data')
    # log_reg.train_model(type='data_splited')

    log_reg.load_all_data_model()
    log_reg.load_splited_data_model()
    training_all_data_loss = log_reg.calculate_training_loss(type='all_data')

    # log_reg.train_model_images()
    log_reg.load_images_data_model()
    # log_reg.save_images_data_model()
    # training_data_images_loss = log_reg.calculate_training_loss_images()

    # log_reg.train_model_images_features()
    log_reg.load_features_images_data_model()
    # training_data_images_features_loss = log_reg.calculate_training_loss_features_images()
    """
    err_train_min = 1
    nbr_compoenents_min = 1
    for i in tqdm.tqdm(range(1, 192)):
        log_reg.load_pca_data(num_comp=i)  # load boath the train & the test
        log_reg.train_pca_model()
        training_pca_data_loss = log_reg.calculate_training_loss_pca_data()
        if training_pca_data_loss < err_train_min:
            err_train_min = training_pca_data_loss
            nbr_compoenents_min = i
    
    
    log_reg.load_pca_data(num_comp=nbr_compoenents_min)
    """

    #log_reg.train_pca_model_cross_validation()
    log_reg.load_pca_data(num_comp=158)
    log_reg.train_pca_model()

    training_pca_data_loss = log_reg.calculate_training_loss_pca_data()

    #print(nbr_compoenents_min)
    #print(err_train_min)
    print(training_all_data_loss)
    print()
    print(training_pca_data_loss)
    #print()
    #print(training_pca_data_loss)
    print()
    log_reg.submit_test_results_pca()
    # print(training_data_images_loss)
    # print()
    # print(training_data_images_features_loss)
    # print()
    # predictions = log_reg.predict_images_data(10)
    # print(predictions)
    # log_reg.submit_test_results()
    # log_reg.submit_test_results_images()
    # log_reg.submit_test_results_images_features()
    # log_reg.submit_test_results_pca()
    # print(training_splited_data_loss)
    # print()
    # print(test_splited_data_loss)
    # print(test_loss)
    # log_reg.save_all_data_model()
    # Test the leaf_image function

    # leaf_id = 341
    # leaf_img = data_man.leaf_image(leaf_id, target_length=60)
    # print(np.shape(leaf_img))
    # df = pd.DataFrame(leaf_img)
    # images_train = data_man.load_images_data_train()
    # print(images_train)
    # print(np.unique(leaf_img))
    # print()
    # print(np.shape(leaf_img))
    # print()
    # plt.imshow(leaf_img, cmap='gray')
    # plt.title('Leaf # ' + str(leaf_id))
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    main()
