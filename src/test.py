import tqdm as tqdm

import models.logistical_regression_model as lr
import data.data_manipulation as dm
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd


# from PIL import Image

def main():
    #data_man = dm.DataManipulation()
    log_reg = lr.LogisticalRegressionModel()

    #log_reg.train_model_images()
    log_reg.load_model_images()
    training_loss_images = log_reg.calculate_training_loss_images()

    #log_reg.train_model_features()
    log_reg.load_model_features()       #############
    training_all_data_loss = log_reg.calculate_training_loss()  #############

    #log_reg.train_model_images_features()
    log_reg.load_model_features_images()
    training_loss_images_features = log_reg.calculate_training_loss_features_images()

    #log_reg.train_model_pca(num_comp=158)
    log_reg.load_model_pca(num_comp=158)
    training_loss_pca = log_reg.calculate_training_loss_pca_data()

    #log_reg.train_pca_model_cross_validation()
    #log_reg.load_pca_data(num_comp=158) #############
    #log_reg.train_pca_model()   #############

    #training_pca_data_loss = log_reg.calculate_training_loss_pca_data() #############

    #print(nbr_compoenents_min)
    #print(err_train_min)
    print("training_loss_model_features: ",training_all_data_loss)
    print()
    print("training_loss_model_images: ",training_loss_images)
    print()
    print("training_loss_model_images_features: ",training_loss_images_features)
    print()
    print("training_loss_model_pca: ",training_loss_pca)
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
