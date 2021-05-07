import numpy as np
import cv2
from PIL import Image
import cv2
import os
import math
import matplotlib.pyplot as plt

wb_plot = False

isic_image_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\ISBI2016_ISIC_Part2B_Training_Data'
aisc_image_path = r'C:\Users\Bruger\AISC_data\images\photodermoscopybaseline-cropped'

def wb_plot(image_path, save_path):
    image_files = os.listdir(image_path)
    list_org_index = [i for i,j in enumerate(image_files) if 'original' in j]
    for i, org_idx in enumerate(list_org_index):
        first_column_index = 6 * i
        if i == 0:
            image_files[i], image_files[org_idx] = image_files[org_idx], image_files[i]
        else:
            image_files[first_column_index], image_files[org_idx] = image_files[org_idx], image_files[first_column_index]
    images = []

    for file in image_files:
        images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))

    image_height, image_width, _ = images[0].shape

    height = int(image_height / 5)
    width = int(image_width / 5 )
    idx = 0
    canvas = np.zeros((height * 4, width * 6, 3))

    for i in range(4):
        for j in range(6):
            print("IDX: " + str(idx))
            resized_image = cv2.resize(images[idx], (width, height), interpolation=cv2.INTER_AREA)
            canvas[i * height:(i+1) * height, j * width:(j+1) * width,:] = np.array(resized_image).astype('uint8')
            idx += 1
    plt.figure()
    plt.imshow(canvas.astype('uint8'))
    # Turn off tick labels
    plt.axis('off')
    plt.show()
    plt.savefig(save_path, format='png', pad_inches=0)


def plot_images(isic_path, aisc_path, number_isic: int, number_aisc: int):
    isic_images_path = os.listdir(isic_path)
    aisc_images_path = os.listdir(aisc_path)
    isic_images_path = isic_images_path[:number_isic]
    aisc_images_path = aisc_images_path[:number_aisc]

    aisc_images = []
    isic_images = []

    for file in isic_images_path:
        isic_images.append(cv2.cvtColor(cv2.imread(isic_path + file), cv2.COLOR_BGR2RGB))

    for file in aisc_images_path:
        aisc_images.append(cv2.cvtColor(cv2.imread(aisc_path + file), cv2.COLOR_BGR2RGB))

    all_images = aisc_images + isic_images

    number_of_columns = 20
    image_width = 1600
    small_width =  int(image_width / number_of_columns)
    small_image_ratio = 16/9
    small_height = int(small_width / small_image_ratio)

    number_isic_rows = math.ceil(number_isic / number_of_columns)
    number_aisc_rows = math.ceil(number_aisc / number_of_columns)
    number_of_rows = math.ceil(number_isic_rows + number_aisc_rows)

    idx = 0
    canvas = np.zeros((small_height * number_of_rows, image_width, 3))

    for i in range(number_of_rows):
        for j in range(number_of_columns):
            print("IDX: " + str(idx))
            if idx < len(all_images):
                resized_image = cv2.resize(all_images[idx], (small_width, small_height), interpolation=cv2.INTER_AREA)
                canvas[i * small_height:(i + 1) * small_height, j * small_width:(j + 1) * small_width, :] = np.array(
                    resized_image).astype('uint8')

            idx += 1
    plt.figure()
    plt.imshow(canvas.astype('uint8'))
    # Turn off tick labels
    plt.axis('off')

def main():
    # image_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\WBAugmented'
    # save_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\\AugmentedImages.png',

    isic_image_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\ISBI2016_ISIC_Part2B_Training_Data\\'
    aisc_image_path = r'C:\Users\Bruger\AISC_data\images\photodermoscopybaseline-cropped\\'

    plot_images(isic_path=isic_image_path, aisc_path=aisc_image_path, number_aisc=200, number_isic=400)
    plt.savefig(r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\pictures\no_cc_pictureplot.png')
    plt.show()

main()