

import pandas as pd
import os
import pickle
import numpy as np
import sys

# first user argument: the path to the augmented images
# second user argument: Name of labels file, full path required
# third user argument: path to the indices pickle, only path needed
# fourth user argument: path where new labels file should be located, name is not required




image_path = sys.argv[1]
labels_path = sys.argv[2]
indices_path = sys.argv[3]
#
# image_path = r'C:\Users\ptrkm\Bachelor\test coloraugmentation\return'
# labels_path = r'C:\Users\ptrkm\Bachelor\test coloraugmentation\label_test_color.csv'
# indices_path = r'C:\Users\ptrkm\Bachelor\test coloraugmentation'


pcl = pickle.load(open(os.path.join(indices_path,'indices_isic2019_one_cv.pkl'), "rb"))
labels = pd.read_csv(labels_path)

image_list = os.listdir(image_path)
image_list = [i.removesuffix('.jpg') for i in image_list]
new_labels =pd.DataFrame()
n_images = len(image_list)
indices_list = []
new_labels_list = []

vals = labels.drop(['image'],axis = 1).values
values = np.ones((1,vals.shape[1]))

for idx, image in enumerate(labels['image']):
    counter = 0
    indexes = []
    for idx_aug, image_aug in enumerate(image_list):
        if image in image_aug:

            new_labels_list.append(image_aug)

            values = np.vstack((values, vals[idx,:].reshape((1,8))))

            indexes.append(idx_aug)
            counter +=1
        if counter == 6:
            for i in sorted(indexes,reverse=True):
                del image_list[i]
            break

values = values[1:,:]

new_labels['image'] = new_labels_list

for j,i in enumerate(labels.columns):
    if 'image' not in i:
        new_labels[i] = values[:,j-1]

new_labels.to_csv(os.path.join(r'C:\Users\ptrkm\Bachelor\test coloraugmentation','labels.csv'),index=False)
train_new = []
val_new = []
for i,j in enumerate(pcl['trainIndCV']):
    image = labels['image'][j]
    counter = 0
    for idx,image_new in enumerate(new_labels['image']):
        if image in image_new:
            train_new.append(idx)
            counter += 1
        if counter == 5:
            break



for i, j in enumerate(pcl['valIndCV']):
    image = labels['image'][j]
    counter = 0
    for idx, image_new in enumerate(new_labels['image']):
        if image in image_new:
            val_new.append(idx)
            counter += 1
        if counter == 5:
            break


pcl_new = {'trainIndCV': np.array(train_new),'valIndCV': np.array(val_new)}

with open(os.path.join(indices_path,'indices_color_augmentation.pkl'),'wb') as handle:
    pickle.dump(pcl_new,handle, protocol=pickle.HIGHEST_PROTOCOL)




















