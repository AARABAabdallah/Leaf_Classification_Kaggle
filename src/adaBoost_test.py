import models.adaBoost_model as ab

def main():
    ada_boost = ab.AdaBoostModel()
    #ada_boost.train_model_cross_validation()
    #ada_boost.train_model_features()
    ada_boost.load_model_features()  #############
    ada_boost.submit_test_results_features()
    training_all_data_loss = ada_boost.calculate_training_loss()  #############
    print("training_loss_model_features: ",training_all_data_loss)


    #ada_boost.train_model_images()
    ada_boost.load_model_images()
    ada_boost.submit_test_results_images()
    training_images_loss = ada_boost.calculate_training_loss_images()
    print("training_loss_model_images: ",training_images_loss)

    #ada_boost.train_model_images_features()
    ada_boost.load_model_features_images()
    ada_boost.submit_test_results_images_features()
    training_features_images_loss = ada_boost.calculate_training_loss_features_images()
    print("training_loss_model_features_images: ", training_features_images_loss)

    #ada_boost.train_model_pca()
    ada_boost.load_model_pca()
    ada_boost.submit_test_results_pca()
    training_pca_loss = ada_boost.calculate_training_loss_pca_data()
    print("training_loss_model_pca: ", training_pca_loss)

if __name__ == "__main__":
    main()
