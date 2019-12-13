import models.gaussianNaiveBayes_model as gnb

def main():
    gnbayes = gnb.GaussianNaiveBayesModel()
    #gnbayes.train_model_features()
    gnbayes.load_model_features()  #############
    gnbayes.submit_test_results_features()
    training_all_data_loss = gnbayes.calculate_training_loss()  #############
    print("training_loss_model_features: ",training_all_data_loss)


    #gnbayes.train_model_images()
    gnbayes.load_model_images()
    gnbayes.submit_test_results_images()
    training_images_loss = gnbayes.calculate_training_loss_images()
    print("training_loss_model_images: ",training_images_loss)

    #gnbayes.train_model_images_features()
    gnbayes.load_model_features_images()
    gnbayes.submit_test_results_images_features()
    training_features_images_loss = gnbayes.calculate_training_loss_features_images()
    print("training_loss_model_features_images: ", training_features_images_loss)

    #gnbayes.train_model_pca_cross_validation()
    gnbayes.load_model_pca()
    gnbayes.submit_test_results_pca()
    training_pca_loss = gnbayes.calculate_training_loss_pca_data()
    print("training_loss_model_pca: ", training_pca_loss)

if __name__ == "__main__":
    main()
