import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.filters import threshold_otsu
from skimage import measure
from scipy import ndimage, signal
import heapq
import color_constancy as cc
import os
import time
import pandas as pd

plt.close('all')

time_zero = time.time()
width = 600
height = 450
preserve_size = 600
paths = [r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_2019_Training_Input\\']
return_folder = r'C:\Users\ptrkm\OneDrive\Dokumenter\TestFolder\return\\'
# paths = [r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\TestRunImages\\']
# return_folder = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\TestRunImagesOutput\\'
standard_size = np.asarray([height, width])
AISC_folders = os.listdir()
preserve_ratio = True
margin = 0.1
crop_black = True
k = 200
threshold = 0.7
resize = True
use_color_constancy = True
write_to_png = False
write = True
ind = 1
all_heights = 0
all_width = 0
use_cropping = False
errors = []
area_threshold = 0.80
def image_preprocess(image):
    if crop_black:
        image = np.uint8(np.array(image))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold_level = threshold_otsu(gray_image)
        gray_image = ndimage.gaussian_filter(gray_image, sigma=np.sqrt(2))
        binary_image = gray_image < threshold_level
        n, m, _ = image.shape

        mean_left = np.mean(image[n // 2 - k // 2:n // 2 + k // 2, :])
        mean_right = np.mean(image[n // 2 - k // 2:n // 2 + k // 2, m - k:])
        mean_top = np.mean(image[:, m // 2 - m // 2:m // 2 + k // 2])
        mean_bottom = np.mean(image[n - k:, m // 2 - m // 2:m // 2 + k // 2])
        mean_middle = np.mean(image[n // 2 - k:n // 2 + k, m // 2 - k:m // 2 + k])

        if mean_middle > np.max([mean_left, mean_top]):
            binary_image = gray_image > threshold_level
        # We now find features in the binarised blobs

        blob_labels = measure.label(binary_image)
        blob_features = measure.regionprops(blob_labels)

        if blob_features:
            largest_blob_idx = np.argmax(np.asarray([blob_features[i].area for i in range(len(blob_features))]))
            largest_blob = blob_features[largest_blob_idx]
            radius = np.mean([largest_blob.major_axis_length, largest_blob.minor_axis_length]) / 2
            equivalent_diameter = largest_blob.equivalent_diameter

            x_min = (largest_blob.centroid[1] - radius + margin * radius).astype(int)
            x_max = (largest_blob.centroid[1] + radius - margin * radius).astype(int)
            y_min = (largest_blob.centroid[0] - radius + margin * radius).astype(int)
            y_max = (largest_blob.centroid[0] + radius - margin * radius).astype(int)
            use_cropping = True
        else:
            use_cropping = False
        if x_min < 0 or x_max > image.shape[1] or y_min < 0 or y_max > image.shape[0]:
            x_center = largest_blob.centroid[1]
            y_center = largest_blob.centroid[0]
            radii = np.arange(0, radius, radius / 20)
            passed = False
            for rad in radii:
                rad = rad.astype(int)
                x_min = (largest_blob.centroid[1] - rad + margin * rad).astype(int)
                x_max = (largest_blob.centroid[1] + rad - margin * rad).astype(int)
                y_min = (largest_blob.centroid[0] - rad + margin * rad).astype(int)
                y_max = (largest_blob.centroid[0] + rad - margin * rad).astype(int)

                if x_min < 0 or x_max > image.shape[1] or y_min < 0 or y_max > image.shape[0]:
                    break
                area_coefficient = np.sum(binary_image[(y_center - rad).astype(int):(y_center + rad).astype(int),
                                          (x_center - rad).astype(int):(x_center + rad).astype(
                                              int)]) / largest_blob.area
                if area_coefficient >= area_threshold:
                    passed = True
                    radius = rad
                    x_min = (largest_blob.centroid[1] - radius + margin * radius).astype(int)
                    x_max = (largest_blob.centroid[1] + radius - margin * radius).astype(int)
                    y_min = (largest_blob.centroid[0] - radius + margin * radius).astype(int)
                    y_max = (largest_blob.centroid[0] + radius - margin * radius).astype(int)
                    use_cropping = True

            if len(blob_features) > 1 and not passed:

                indices = np.where(np.arange(len(blob_features)) != largest_blob_idx)[0].astype(int)
                without_largest = [blob_features[idx] for idx in indices]
                second_largest_idx = np.argmax(
                    np.asarray([without_largest[i].area for i in range(len(without_largest))]))
                second_largest = without_largest[second_largest_idx]
                radius = np.mean([second_largest.major_axis_length, second_largest.minor_axis_length]) / 2

                x_min = (second_largest.centroid[1] - radius + margin * radius).astype(int)
                x_max = (second_largest.centroid[1] + radius - margin * radius).astype(int)
                y_min = (second_largest.centroid[0] - radius + margin * radius).astype(int)
                y_max = (second_largest.centroid[0] + radius - margin * radius).astype(int)

                if x_min < 0 or x_max > image.shape[1] or y_min < 0 or y_max > image.shape[0]:
                    use_cropping = False
        else:
            use_cropping = True
        if use_cropping:

            mean_inside = np.mean(image[y_min:y_max, x_min:x_max, :])
            exclude_x = np.ones(image.shape[1], dtype=int)
            exclude_y = np.ones(image.shape[0], dtype=int)

            mean_outside = (np.mean(image[:y_min, :, :]) + np.mean(image[y_min:y_max, :x_min, :]) +
                            np.mean(image[y_max:, :, :]) + np.mean(image[y_min:y_max, x_max:, :])) / 4

            if mean_outside / mean_inside > 0.3:
                use_cropping = False

        if use_cropping:
            image = image[y_min:y_max, x_min:x_max, :]
        if image.shape[0] > 0 and image.shape[1] > 0 and image.shape[2] > 0:
            if resize:
                if preserve_ratio:
                    if image.shape[0] > image.shape[1]:
                        image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])

                    if image.shape[1] != preserve_size:
                        ratio = preserve_size / image.shape[1]
                        image = cv2.resize(image, dsize=(round(image.shape[0] * ratio), preserve_size))
                else:
                    if image.shape[0] > image.shape[1]:
                        image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])
                    if image.shape[0] != standard_size[0] or image.shape[1] != standard_size[1]:
                        image = cv2.resize(image, dsize=(standard_size[0], standard_size[1]))
            if use_color_constancy:
                color_constancy = cc.general_color_constancy()
                new_image = color_constancy.color_augment(image)
                new_image = np.uint8(new_image)

            else:
                new_image = image

            return Image.fromarray(new_image)