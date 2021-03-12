
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
import color_constancy as cc


width = 600
height = 450
preserve_size = 600
paths = [r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_2019_Test_Input\\']
return_folder = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_test_cropped\\'
# paths = [r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\TestRunImages\\']
# return_folder = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\TestRunImagesOutput\\'
standard_size = np.asarray([height, width])
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
import shutil
full_data = os.listdir(paths[0])
cropped_data = os.listdir(return_folder)

unused_data = list(set(full_data)-set(cropped_data))
idx = 0

for i in unused_data:
    if 'txt' not in i:
        try:
            image = cv2.imread(paths[0]+i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(i)

        if resize:
            if preserve_ratio:
                if image.shape[0] > image.shape[1]:
                    image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])

                if image.shape[1] != preserve_size:
                    ratio = preserve_size / image.shape[1]
                    try:
                        image = cv2.resize(image, dsize=(round(image.shape[0] * ratio), preserve_size))
                    except:
                        print("resize problem on image" + images)
                        errors.append(images)
                        continue
        R, G, B, new_image = cc.general_color_constancy(image, 0, 6, 0)
        new_image = np.uint8(new_image)

        im = Image.fromarray(new_image).convert('RGB')
        im.save(return_folder + i)

    idx += 1

    print(idx)



for i, images in enumerate(cropped_data):
    if 'txt' not in images:
        try:
            image = cv2.imread(return_folder+images)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[0] < 100 or image.shape[1] < 100:
                image = cv2.imread(paths[0]+images)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if resize:
                    if preserve_ratio:
                        if image.shape[0] > image.shape[1]:
                            image = np.moveaxis(image, [0, 1, 2], [1, 0, 2])

                        if image.shape[1] != preserve_size:
                            ratio = preserve_size / image.shape[1]
                            try:
                                image = cv2.resize(image, dsize=(round(image.shape[0] * ratio), preserve_size))
                            except:
                                print("resize problem on image" + images)
                                errors.append(images)
                                continue
                R, G, B, new_image = cc.general_color_constancy(image, 0, 6, 0)
                new_image = np.uint8(new_image)

                im = Image.fromarray(new_image).convert('RGB')
                im.save(return_folder + images)
        except:
            print(images)
            continue

        if i % 100 == 0: print(i)


