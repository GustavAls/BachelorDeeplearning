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

plt.close('all')
width = 600
height = 450
preserve_size = 600
paths = [r'C:\Users\ptrkm\OneDrive\Dokumenter\TestFolder\\']
return_folder = r'C:\Users\ptrkm\OneDrive\Dokumenter\TestFolder\return\\'
standard_size = np.asarray([height, width])
preserve_ratio = True
margin = 0.1
crop_black = True
threshold = 0.3
resize = False
use_color_constancy = False
write_to_png = False
write = True
ind = 1
all_heights = 0
all_width = 0
use_cropping = False
errors = []
for i, j in enumerate(os.listdir(paths[0])):
    if i == 0:
        try:
            image = cv2.imread(paths[0]+j)
            print("yes man")
        except:
            print("File " + j + "Could not read :(")
            errors.append(j)
            continue
        print("hej")
        print(j)
        if crop_black:

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_level = threshold_otsu(gray_image)
            gray_image = ndimage.gaussian_filter(gray_image, sigma=np.sqrt(2))
            binary_image = gray_image < threshold_level

            # We now find features in the binarised blobs

            blob_labels = measure.label(binary_image)
            blob_features = measure.regionprops(blob_labels)
            if blob_features:
                largest_blob_idx = np.argmax(np.asarray([blob_features[i].area for i in range(len(blob_features))]))
                largest_blob = blob_features[largest_blob_idx]
                radius = np.mean([largest_blob.major_axis_length, largest_blob.minor_axis_length]) / 2

                x_min = (largest_blob.centroid[1] - radius + margin * radius).astype(int)
                x_max = (largest_blob.centroid[1] + radius - margin * radius).astype(int)
                y_min = (largest_blob.centroid[0] - radius + margin * radius).astype(int)
                y_max = (largest_blob.centroid[0] + radius - margin * radius).astype(int)


                use_cropping = True
            else:
                use_cropping = False

            if x_min < 0 or x_max > image.shape[1] or y_min < 0 or y_max > image.shape[0]:
                if len(blob_features) > 1:
                    without_largest = blob_features[np.arange(len(blob_features)) != largest_blob_idx]
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
                exclude_x = np.ones(image.shape[1],dtype=int)
                exclude_y = np.ones(image.shape[0],dtype=int)

                mean_outside = (np.mean(image[:y_min,:,:])+np.mean(image[y_min:y_max,:x_min,:])+
                                np.mean(image[y_max:,:,:])+np.mean(image[y_min:y_max,x_max:,:]))/4




                if mean_outside / mean_inside < threshold:
                    use_cropping = False
            if use_cropping:
                image = image[y_min:y_max, x_min:x_max, :]
            if resize:
                if preserve_ratio:
                    if image.shape[0] > image.shape[1]:
                        image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])

                    if image.shape[1] != preserve_size:
                        ratio = preserve_size / image.shape[1]

                        image = cv2.resize(image, dsize=[(round(image.shape[0] * ratio)).astype(int), preserve_size])
                else:
                    if image.shape[0] > image.shape[1]:
                        image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])
                    if image.shape[0] != standard_size[0] or image.shape[1] != standard_size[1]:
                        image = cv2.resize(image, dsize=[standard_size])
            if use_color_constancy:

                R, G, B, new_image = cc.general_color_constancy(image, 0, 6, 0)
                new_image = np.uint8(new_image)
            else:
                new_image = image

            if write:
                if write_to_png:
                    im = Image.fromarray(new_image.astype('uint8')).convert('RGB')
                    im.save(return_folder + j.name.replace('.jpg', '.png'))
                else:
                    im = Image.fromarray(new_image.astype('uint8')).convert('RGB')
                    im.save(return_folder + j)
            if i % 1000: print(i)
