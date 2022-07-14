import matplotlib.pyplot as plt
import math
import cv2
from skimage import img_as_float
import matplotlib.patches as matlab_patches


class Utils:
    @staticmethod
    def display_image(image, title):
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title(title)
        plt.show()

    @staticmethod
    def __confusion_matrix_from_expert_mask(image, expert_mask):
        result = []
        tests = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0
        }

        for i in range(len(image)):
            row = []
            for j in range(len(image[i])):
                if image[i][j] == 1.0:
                    if expert_mask[i][j] == 1.0:
                        row.append([0, 255, 0])
                        tests['true_positives'] += 1
                    elif expert_mask[i][j] == 0.0:
                        row.append([255, 0, 0])
                        tests['false_positives'] += 1
                elif image[i][j] == 0.0:
                    if expert_mask[i][j] == 1.0:
                        row.append([0, 0, 255])
                        tests['false_negatives'] += 1
                    elif expert_mask[i][j] == 0.0:
                        row.append([0, 0, 0])
                        tests['true_negatives'] += 1
            result.append(row)
        return result, tests

    @staticmethod
    def display_visualization(computed_mask, expert_mask):
        expert_mask_image = img_as_float(cv2.imread(expert_mask, cv2.IMREAD_GRAYSCALE));
        res, tests = Utils.__confusion_matrix_from_expert_mask(computed_mask, expert_mask_image)

        accuracy = (tests['true_positives'] + tests['true_negatives']) / (tests['true_negatives'] + tests['false_negatives'] + tests['true_positives'] + tests['false_positives'])
        sensitivity = tests['true_positives'] / (tests['true_positives'] + tests['false_negatives'])
        specificity = tests['true_negatives'] / (tests['false_positives'] + tests['true_negatives'])
        average = (specificity + sensitivity) / 2
        geometrical_average = math.sqrt(specificity * sensitivity)

        print("trafność:", accuracy)
        print("czułość:", sensitivity)
        print("swoistość:", specificity)
        print("średnia arytmetyczna czułości i swoistości:", average)
        print("średnia geometryczna czułości i swoistości:", geometrical_average)

        black_path = matlab_patches.Patch(color='black', label='TN')
        green_patch = matlab_patches.Patch(color='green', label='TP')
        red_patch = matlab_patches.Patch(color='red', label='FP')
        blue_patch = matlab_patches.Patch(color='blue', label='FN')
        plt.legend(handles=[red_patch, blue_patch, black_path, green_patch], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.imshow(res)
        plt.title('Algorithm evaluation')
        plt.show()
