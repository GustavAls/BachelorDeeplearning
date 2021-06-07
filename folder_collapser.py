
import os
import pandas as pd
import pickle
import numpy as np
import sys
import shutil
top_folder = r'C:\Users\Bruger\AISC_data\images\\'
cropped_folder = r'C:\Users\Bruger\PycharmProjects\Bachelor\AISC_images\\'
return_folder = r'C:\Users\Bruger\PycharmProjects\Bachelor\AISC_uncropped\\'

for folder in os.listdir(top_folder):
    for idx, image in enumerate(os.listdir(top_folder + folder)):
        list_dir = os.listdir(cropped_folder)
        image_name = image[:-5] + ".jpg"
        if image_name in list_dir:
            shutil.copyfile(os.path.join(top_folder + folder,image), os.path.join(return_folder,image))

        if idx % 100 == 0:
            print(idx)


