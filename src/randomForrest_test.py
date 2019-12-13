import models.randomForrest_model as rf

def main():
    rand_forrest = rf.RandomForrestModel()
    #rand_forrest.train_model_cross_validation()
    #rand_forrest.train_model_features()
    rand_forrest.load_model_features()  #############
    #rand_forrest.submit_test_results_features()
    training_all_data_loss = rand_forrest.calculate_training_loss()  #############
    print("training_loss_model_features: ",training_all_data_loss)


    #rand_forrest.train_model_images()
    rand_forrest.load_model_images()
    rand_forrest.submit_test_results_images()
    training_images_loss = rand_forrest.calculate_training_loss_images()
    print("training_loss_model_images: ",training_images_loss)


    #rand_forrest.train_model_images_features()
    rand_forrest.load_model_features_images()
    rand_forrest.submit_test_results_images_features()
    training_features_images_loss = rand_forrest.calculate_training_loss_features_images()
    print("training_loss_model_features_images: ", training_features_images_loss)

    #rand_forrest.train_model_pca()
    #rand_forrest.train_model_pca_cross_validation()
    rand_forrest.load_model_pca()
    #rand_forrest.submit_test_results_pca()
    training_pca_loss = rand_forrest.calculate_training_loss_pca_data()
    print("training_loss_model_pca: ", training_pca_loss)

if __name__ == "__main__":
    main()
