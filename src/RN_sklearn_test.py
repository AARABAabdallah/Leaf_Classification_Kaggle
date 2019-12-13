import models.RN_sklearn_model as RN_skl


def main():

    rn_skl = RN_skl.RN_sklearn_model()

    rn_skl.load_model_features_images()
    print(rn_skl.get_training_loss_features_images())
    # rn_skl.features_cross_valid(layer2_range=[80,100,140,160],layer3_range=range(0),layer1_range=range(10,241,5))
    # print(rn_skl.get_training_loss())

    rn_skl.submit_test_results_images_features()

if __name__ == "__main__":
    main()
