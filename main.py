import cv2
import pandas
from BasicDetection import BasicDetection
from Utils import Utils
from PrepareData import PrepareData
from Classifier import Classifier


prepare_data = PrepareData()


def get_training_data():
    frames = []
    training_images = []

    for i in range(1, 5):
        number = ("%02d" % (i,))
        training_images.append([f"healthy/{number}_h.jpg", f"healthy_manualsegm/{number}_h.tif"])

    for image_data in training_images:
        df = prepare_data.get_labeled_data(image_data, 5)
        df = prepare_data.down_sample(df)
        frames.append(df)

    return pandas.concat(frames)


def get_basic_detection_results():
    for i in range(6, 11):
        number = ("%02d" % (i,))
        chosen_image = F"{number}_h"
        basic_detection = BasicDetection(f"healthy/{chosen_image}.jpg")
        computed_mask = basic_detection.compute_expert_mask()

        Utils.display_image(computed_mask, F"Computed mask {chosen_image}")
        print(F"IMAGE: {chosen_image}")
        Utils.display_visualization(computed_mask, f"healthy_manualsegm/{chosen_image}.tif")


def get_classifier_results():
    training_df = get_training_data()

    classifier = Classifier(training_df.loc[:, training_df.columns != 'label'], training_df.loc[:, 'label'])
    classifier.fit()
    classifier.score()

    for i in range(6, 11):
        number = ("%02d" % (i,))
        chosen_image = F"{number}_h"

        print(F"IMAGE: {chosen_image}")
        img_data = [F"healthy/{chosen_image}.jpg", F"healthy_manualsegm/{chosen_image}.tif"]

        full_img_df = prepare_data.get_labeled_data(img_data, 1)
        print("prepare")
        labels = classifier.predict(full_img_df.loc[:, full_img_df.columns != 'label'])
        print("labels")
        full_img_labels = PrepareData.normalize(cv2.imread(img_data[1], cv2.IMREAD_GRAYSCALE))
        print("full image labels")
        Utils.display_image(cv2.imread(img_data[0], cv2.IMREAD_GRAYSCALE), F"Original image {chosen_image}")
        detected_image = prepare_data.reshape_labels(labels, full_img_labels.shape)
        Utils.display_image(detected_image, F"Detected vessels {chosen_image}")
        Utils.display_visualization(detected_image, F'healthy_manualsegm/{chosen_image}.tif')


if __name__ == '__main__':
    # get_basic_detection_results()
    get_classifier_results()




