import numpy as np
import pandas as pd
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import psutil
import matplotlib.pyplot as plt
import os
import pickle
import sklearn.metrics as metrics
from scipy import io

os.chdir(r'C:\Users\ptrkm\Bachelor')

pcl = pickle.load(open(r'indices_aisc_plus_isic.pkl', "rb"))

labels = pd.read_csv(r'labels_aisc_isic.csv')
labels_our = pd.read_csv(r'labels.csv')

img_uids_train = pd.read_csv(r'AISC_train_im_paths.txt',header = None)
img_uids_val = pd.read_csv(r'AISC_val_im_paths.txt', header = None)


validation = []
training = []

for i in img_uids_val[0]:
    if i in img_uids_train:
        breakpoint()




counter = 0

for i in range(len(pcl['trainIndCV'])):
    if i == 3:
        train_inds = pcl['trainIndCV'][i]
        val_inds = pcl['valIndCV'][i]

        training = [j for j,i in enumerate(labels['image']) if 'ISIC' in i and j in train_inds]
        validation = [j for j,i in enumerate(labels['image']) if 'ISIC' in i and j in val_inds]
        num_not = 0

        for idx, j_uids in enumerate(labels['image']):
            if 'ISIC' not in j_uids:
                for im_paths in img_uids_train[0]:

                    if j_uids in im_paths:
                        training.append(idx)

                        break
                for im_paths in img_uids_val[0]:
                    if j_uids in im_paths:
                        validation.append(idx)
                        break
                if training[-1] == validation[-1]:
                    breakpoint()

        indices_new = {'trainIndCV':np.array(training),'valIndCV':np.array(validation)}
        with open('indices_aisc_plus_isic.pkl', 'wb') as handle:
            pickle.dump(indices_new,handle, protocol=pickle.HIGHEST_PROTOCOL)





