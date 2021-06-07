import numpy as np
import cv2
from PIL import Image, ImageOps
import cv2
import os
import pandas as pd
import math
import torch
import random
import utils
from random import sample
from torchvision import transforms
from torchvision.transforms import functional
from auto_augment_for_plotting import AutoAugment
from auto_augment import AutoAugment as AutoAugmentOriginal
from color_constancy import general_color_constancy
import image_preprocess_one_image as preprocess
from utils import Cutout_v0 as CutOut_v0_Original
import matplotlib.pyplot as plt

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


def image_cropping(image_as_array):
    ''':cvar
    Return: PIL image cropped to standard
    '''
    image_height, image_width = image_as_array.shape[0], image_as_array.shape[1]

    preserve_width = 600
    preserve_height = 450
    ratio = preserve_width / image_width
    new_height = round(image_height * ratio)

    if new_height == 450:
        resized_image = np.array(cv2.resize(image_as_array,
                                            dsize=(preserve_width, preserve_height)))
    elif new_height > 450:
        aspect_ratio = image_width / image_height
        set_height = round(600 * aspect_ratio)

        image_as_array = np.array(cv2.resize(image_as_array,
                                             dsize=(preserve_width, set_height)))

        image_as_array = image_as_array[set_height // 2 - 225: set_height // 2 + 225, :, :]
        resized_image = Image.fromarray(image_as_array)

    else:
        aspect_ratio = image_width / image_height
        set_width = round(450 * aspect_ratio)

        image_as_array = np.array(cv2.resize(image_as_array,
                                             dsize=(set_width, preserve_height)))

        image_as_array = image_as_array[:, set_width // 2 - 300: set_width // 2 + 300, :]
        resized_image = Image.fromarray(image_as_array)
    return resized_image


def plot_images(isic_path, aisc_path, isic_cropped_path, aisc_cropped_path, number_isic: int, number_aisc: int):
    isic_images_path = os.listdir(isic_path)
    aisc_cropped_images_path = os.listdir(aisc_cropped_path)

    random.seed(43)
    isic_images_path = sample(isic_images_path, number_isic)
    aisc_cropped_images_path = sample(aisc_cropped_images_path, number_aisc)

    names = {}
    names['all_images_names'] = isic_images_path + aisc_cropped_images_path
    path = r'C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures'
    pd.DataFrame.from_dict(names).to_csv(os.path.join(path,r'all_names.csv'))

    aisc_images = []
    aisc_cc_images = []

    isic_images = []
    isic_cc_images = []

    for file in aisc_cropped_images_path:

        ###########################

        filename = file[:-4] + ".jpeg"
        loaded_image = Image.open(aisc_path + "\\" + filename)
        preprocessed_image = Image.open(aisc_cropped_path + "\\" + file)

        preprocessed_image = np.array(preprocessed_image)

        preprocessed_image = image_cropping(preprocessed_image)

        aisc_cc_images.append(preprocessed_image)

        image_as_array = np.array(loaded_image)

        resized_image = image_cropping(image_as_array)

        aisc_images.append(resized_image)

    for file in isic_images_path:
        loaded_image = Image.open(isic_path + "\\" + file)
        preprocessed_image = Image.open(isic_cropped_path + "\\" + file)

        preprocessed_image = np.array(preprocessed_image)

        preprocessed_image = image_cropping(preprocessed_image)

        isic_cc_images.append(preprocessed_image)

        image_as_array = np.array(loaded_image)

        resized_image = image_cropping(image_as_array)

        isic_images.append(resized_image)

    all_images = aisc_images + isic_images
    all_cc_images = aisc_cc_images + isic_cc_images

    number_of_columns = 60
    image_width = 2500
    small_width =  int(image_width / number_of_columns)
    small_image_ratio = 600 / 450
    small_height = int(small_width / small_image_ratio)

    number_isic_rows = math.ceil(number_isic / number_of_columns)
    number_aisc_rows = math.ceil(number_aisc / number_of_columns)
    number_of_rows = math.ceil(number_isic_rows + number_aisc_rows)

    idx = 0
    canvas = np.zeros((small_height * number_of_rows, image_width, 3))
    canvas_cc = np.zeros((small_height * number_of_rows, image_width, 3))
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            if idx < len(all_images):
                image_as_array = np.array(all_images[idx])

                resized_image = np.array(cv2.resize(image_as_array,
                                                    dsize=(small_width, small_height)))
                canvas[i * small_height:(i + 1) * small_height, j * small_width:(j + 1) * small_width, :] = np.array(
                    resized_image).astype('uint8')

                image_as_array = np.array(all_cc_images[idx])

                resized_image = np.array(cv2.resize(image_as_array,
                                                    dsize=(small_width, small_height)))
                canvas_cc[i * small_height:(i + 1) * small_height, j * small_width:(j + 1) * small_width, :] = np.array(
                    resized_image).astype('uint8')

            idx += 1
    canvas = Image.fromarray(np.uint8(canvas[:,:int((59 / 60) * image_width),:]))
    canvas_cc = Image.fromarray(np.uint8(canvas_cc[:,:int((59 / 60) * image_width),:]))
    canvas.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\no_color_constancy.jpg")
    canvas_cc.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\with_color_constancy.jpg")





class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, x, y):
        self.n_holes = n_holes
        self.length = length
        self.x = x
        self.y = y

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        #print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):

            y1 = np.clip(self.y - self.length // 2, 0, h)
            y2 = np.clip(self.y + self.length // 2, 0, h)
            x1 = np.clip(self.x - self.length // 2, 0, w)
            x2 = np.clip(self.x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask,axis=2)
        img = Image.fromarray(img)
        return img


def image_data_augmenter(image, cutoutCordinates, randrange, autoAugment = False):
    full_rot = 180
    scale = (0.8, 1.2)
    shear = 10
    cutout = int(16 * (600 / 224))
    all_transforms = []

    if autoAugment:
        # all_transforms.append(AutoAugment(randrange=randrange))
        all_transforms.append(AutoAugmentOriginal())
    else:
        #Random flips
        all_transforms.append(transforms.RandomHorizontalFlip())
        all_transforms.append(transforms.RandomVerticalFlip())
        # Full rot
        all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(full_rot,
                                                                               scale=scale,
                                                                               shear=shear,
                                                                               resample=Image.NEAREST),
                                                       transforms.RandomAffine(full_rot,
                                                                               scale=scale,
                                                                               shear=shear,
                                                                               resample=Image.BICUBIC),
                                                       transforms.RandomAffine(full_rot,
                                                                               scale=scale,
                                                                               shear=shear,
                                                                               resample=Image.BILINEAR)]))
        #Color jitter
        all_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))

        # Cutout

        all_transforms.append(Cutout_v0(n_holes=1, length=cutout, x=cutoutCordinates[0], y=cutoutCordinates[1]))
        #all_transforms.append(utils.Cutout_v0(n_holes=1, length=cutout))



    # All transforms
    composed = transforms.Compose(all_transforms)
    return composed(image)

def deterministic_daisy_lab_image_data_augmenter(image, cutoutCordinates, scale, shear, rotate, brightness,
                                                 saturation,horizontalFlip=False, verticalFlip = False):

    cutout = int(16 * (600 / 224))

    cutout = Cutout_v0(n_holes=1, length=cutout, x=cutoutCordinates[0], y=cutoutCordinates[1])

    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    vertical_flip =  transforms.RandomVerticalFlip(p=1.0)

    if horizontalFlip: image = horizontal_flip(image)

    if verticalFlip: image = vertical_flip(image)

    image = functional.affine(image, angle=0, translate=[0, 0], shear=[0.0], scale=scale)

    image = functional.affine(image, angle=0, translate=[0, 0], shear=shear, scale=1)

    image = functional.affine(image, angle=rotate, translate=[0, 0], shear=[0.0], scale=1,
                                              resample=Image.BILINEAR)

    image = functional.adjust_brightness(image, brightness_factor=brightness)

    image = functional.adjust_saturation(image, saturation_factor=saturation)

    image = cutout(image)


    return image

def plot_augmented_images(image_path, number_of_images: int, number_of_augmentations: int, autoAugment = False):
    images = os.listdir(image_path)
    #Random sampler
    # images = sample(images, number_of_images)
    final_images = []

    #Generate cutout positions
    cutout = {}
    for i in range(number_of_augmentations):
        cutout[i] = (random.randrange(100, 400), random.randrange(100, 400))

    for image in images:
        loaded_image = Image.open(image_path + "\\" + image)
        loaded_image = loaded_image.resize(size=(600, 450), resample=Image.BILINEAR)
        final_images.append(loaded_image)
        for i in range(number_of_augmentations):
            torch.manual_seed(i)
            final_images.append(image_data_augmenter(loaded_image, cutout[i], i, autoAugment=autoAugment))

    number_of_rows = number_of_augmentations + 1
    number_of_columns = number_of_images
    aspect_ratio = (600 / 450) / (number_of_rows / number_of_columns)
    image_width = 1600
    image_height = int(1600 / aspect_ratio)
    small_width = int(image_width / number_of_columns)
    small_height = int(image_height / number_of_rows)
    idx = 0
    canvas = np.zeros((small_height * number_of_rows, image_width, 3))

    for i in range(number_of_columns):
        for j in range(number_of_rows):
            print("IDX: " + str(idx))
            resized_image = final_images[idx].resize(size=(small_width, small_height), resample=Image.BILINEAR)
            canvas[j * small_height:(j + 1) * small_height, i * small_width:(i + 1) * small_width, :] = np.array(
                resized_image).astype('uint8')

            idx += 1
    plt.figure()
    plt.imshow(canvas.astype('uint8'))
    # Turn off tick labels
    plt.axis('off')

def plot_augmented_subplots(image_path, autoAugment = False):
    images = os.listdir(image_path)
    final_images = []

    scale = [0.82, 1.14, 0.89, 1.01]
    shear = [-9.1, 4.1, 0.9, 8.3]
    rotate = [80, -10, 35, 112]
    brightness = [244. / 255, 230. / 255., 275. / 255., 280. / 255.]
    saturation = [1.2, 0.79, 1.4, 0.9]
    cutout = [[244, 244], [120, 300], [125, 90], [320, 300]]
    horizontal_flip = [False, True, False, True]
    vertical_flip = [False, False, False, True]


    #Generate cutout positions

    for idx,image in enumerate(images):
        loaded_image = Image.open(image_path + "\\" + image)
        loaded_image = loaded_image.resize(size=(600, 450), resample=Image.BILINEAR)
        final_images.append(loaded_image)
        for i in range(3):
            if autoAugment:
                if idx == 0:
                    do_print = True
                else:
                    do_print = False
                AutoAug = AutoAugment(randrange=i + 11)
                final_images.append(AutoAug(loaded_image, do_print=do_print))
            else:
                final_images.append(deterministic_daisy_lab_image_data_augmenter(loaded_image,cutout[i], scale[i], shear[i], rotate[i],
                                                                             brightness[i], saturation[i], horizontal_flip[i], vertical_flip[i]))

    fig, axes = plt.subplots(4, 8, constrained_layout=True)
    idx = 0
    diagnosis_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    for j in range(8):
        for i in range(4):
            axes[i, j].imshow(final_images[idx])
            axes[i, j].axis('off')
            idx += 1
    fig.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.1,
                        hspace= -0.74)
    plt.show()

def plot_wb_augs(image_path):
    images = os.listdir(image_path)
    final_images = []
    for image in images:
        loaded_image = Image.open(image_path + "\\" + image)
        loaded_image = np.array(loaded_image.resize(size=(450, 450), resample=Image.BILINEAR))
        final_images.append(loaded_image)

    fig, axes = plt.subplots(4, 6, constrained_layout=True)
    idx = 0

    for i in range(4):
        for j in range(6):
            axes[i, j].imshow(final_images[idx])
            axes[i, j].axis('off')
            idx += 1
    fig.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.1,
                        hspace=0)
    plt.show()


def melanoma_image_aug(image_path):
    '''
    Transformations:
    horizontal flip
    vertical flip
    scale = 0.9
    shear = 8.0
    rotation = 150.0
    brightness factor = 223 / 255
    saturation factor = 0.5
    '''
    # full_rot = 180
    # scale = (0.8, 1.2)
    # shear = 10
    cutout = int(16 * (600 / 224))
    cutout_coordinate = [100,200]

    melanoma_image = Image.open(image_path)

    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    # vertical_flip = transforms.RandomVerticalFlip(p=1.0)
    # Cutout
    cutout = Cutout_v0(n_holes=1, length=cutout, x=cutout_coordinate[0], y=cutout_coordinate[1])

    all_transforms_melanoma_image = melanoma_image

    horizontal_flip_melanoma_image = horizontal_flip(melanoma_image)
    all_transforms_melanoma_image = horizontal_flip(all_transforms_melanoma_image)

    # vertical_flip_melanoma_image = vertical_flip(melanoma_image)
    # all_transforms_melanoma_image = vertical_flip(all_transforms_melanoma_image)

    scale_melanoma_image = functional.affine(melanoma_image, angle=0, translate=[0,0], shear=[0.0], scale = 0.9)
    all_transforms_melanoma_image = functional.affine(all_transforms_melanoma_image, angle=0, translate=[0,0], shear=[0.0], scale = 0.9)

    shear_melanoma_image = functional.affine(melanoma_image, angle=0, translate=[0,0], shear=[8.0], scale = 1)
    all_transforms_melanoma_image = functional.affine(all_transforms_melanoma_image, angle=0, translate=[0,0], shear=[8.0], scale = 1)

    rotate_melanoma_image = functional.affine(melanoma_image, angle=150, translate=[0,0], shear=[0.0], scale = 1,resample=Image.BILINEAR)
    all_transforms_melanoma_image = functional.affine(all_transforms_melanoma_image, angle=150, translate=[0,0], shear=[0.0], scale = 1, resample=Image.BILINEAR)

    brightness_melanoma_image = functional.adjust_brightness(melanoma_image, brightness_factor=1 - 32. / 255.)
    all_transforms_melanoma_image = functional.adjust_brightness(all_transforms_melanoma_image, brightness_factor=1 - 32. / 255.)

    saturation_melanoma_image = functional.adjust_saturation(melanoma_image, saturation_factor=0.5)
    all_transforms_melanoma_image = functional.adjust_saturation(all_transforms_melanoma_image, saturation_factor=0.5)

    cutout_melanoma_image = cutout(melanoma_image)
    all_transforms_melanoma_image = cutout(all_transforms_melanoma_image)


    melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\orginial.jpg")
    horizontal_flip_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\horizontal_flip.jpg")
    # vertical_flip_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\vertical_flip.jpg")
    scale_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\scale.jpg")
    shear_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\shear.jpg")
    rotate_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\rotate.jpg")
    brightness_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\brightness.jpg")
    saturation_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\saturation.jpg")
    cutout_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\cutout.jpg")

    all_transforms_melanoma_image.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\all_transforms.jpg")
    random.seed(42)

    scale = random.uniform(0.8, 1.2)
    shear = random.uniform(-10, 10)
    rotate = random.uniform(-180, 180)
    brightness = random.uniform(1 - 32. / 255., 1 + 32. / 255.)
    saturation = random.uniform(0.5, 1.5)
    cutout_coordiante = [random.randrange(100, 400), random.randrange(100, 400)]

    print("Rand 1")
    print("Scale " + str(scale))
    print("Shear " + str(shear))
    print("Rotate " + str(rotate))
    print("Brigthness " + str(brightness))
    print("Saturation " + str(saturation))
    print("Cutout coordinates" + str(cutout_coordiante))


    all_transforms_1 = deterministic_daisy_lab_image_data_augmenter(melanoma_image, cutoutCordinates=cutout_coordiante, scale=scale,
                                                                    shear=shear, rotate=rotate, brightness=brightness,
                                                                    saturation=saturation, horizontalFlip=True)

    all_transforms_1.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\rand_1.jpg")

    scale = random.uniform(0.8, 1.2)
    shear = random.uniform(-10, 10)
    rotate = random.uniform(-180, 180)
    brightness = random.uniform(1 - 32. / 255., 1 + 32. / 255.)
    saturation = random.uniform(0.5, 1.5)
    cutout_coordiante = [random.randrange(100, 400), random.randrange(100, 400)]

    print("Rand 2")
    print("Scale " + str(scale))
    print("Shear " + str(shear))
    print("Rotate " + str(rotate))
    print("Brigthness " + str(brightness))
    print("Saturation " + str(saturation))
    print("Cutout " + str(cutout_coordiante))

    all_transforms_2 = deterministic_daisy_lab_image_data_augmenter(melanoma_image,cutoutCordinates=cutout_coordiante, scale=scale,
                                                                    shear=shear, rotate=rotate, brightness=brightness,
                                                                    saturation=saturation, horizontalFlip=False)

    all_transforms_2.save(r"C:\Users\Bruger\PycharmProjects\Bachelor\Article_figures\rand_2.jpg")

def main():

    #Plotting
    # isic_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\2019_data\\'
    # aisc_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\AISC_uncropped\\'
    # isic_cropped_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\2019_cropped_data\\'
    # aisc_cropped_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\AISC_images\\'
    #
    # plot_images(isic_path=isic_path, aisc_path=aisc_path,
    #             isic_cropped_path=isic_cropped_path, aisc_cropped_path=aisc_cropped_path, number_aisc=1800, number_isic=1800)

    #Plotting data augmentations
    # isic_image_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\ISIC2019_plotting_images_2\\'
    # plot_augmented_subplots(isic_image_path, autoAugment=True)

    isic_image_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\WBAugmented\\'
    plot_wb_augs(isic_image_path)
    breakpoint()

    # isic_image_path = r'C:\Users\Bruger\PycharmProjects\Bachelor\2019_cropped_data\ISIC_0000022_downsampled.jpg'
    # melanoma_image_aug(isic_image_path)

    # plt.show()

main()