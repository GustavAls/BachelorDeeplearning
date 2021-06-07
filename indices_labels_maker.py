import numpy as np
import pandas as pd
import pickle
import os

val_set = pickle.load(open(r'/home/s184400/2019_val_set_new.pkl', "rb"))
train_set = pickle.load(open(r'/home/s184400/2019_train_set_new.pkl', 'rb'))

image_path = r'/scratch/s184400/AISC_full/official/'
isic_labels_path = r'/home/s184400/isic2019/labels/official/ISIC_2019_Training_GroundTruth.csv'

new_indices_path = r'/home/s184400/indices_aisc_full.pkl'
new_labels_path = r'/scratch/s184400/AISC_full_labels/official/aisc_full_labels.csv'

images = os.listdir(image_path)
indices_dict = {}
indices_dict['trainIndCV'] = []
indices_dict['valIndCV'] = []

for idx, image in enumerate(images):
    print(image)
    if 'ISIC' in image:
        indices_dict['trainIndCV'].append(idx)
    elif image in train_set['paths']:
        indices_dict['trainIndCV'].append(idx)
        print("Success train")
    elif image in val_set['paths']:
        indices_dict['valIndCV'].append(idx)
        print("Success val")
    else:
        print("Image name is not in CV split")


with open(new_indices_path, 'wb') as f:
    pickle.dump(indices_dict, f, pickle.HIGHEST_PROTOCOL)


old_labels_dict = {'image': train_set['paths'] + val_set['paths'], 'labels': train_set['labels'] + val_set['labels']}
isic_labels = pd.read_csv(isic_labels_path)
isic_labels_array = np.array(isic_labels)
aisc_labels_array = np.zeros((len(images) - 25331, 10))

for idx, image in enumerate(images):
    if 'ISIC' not in image:
        for j, name in enumerate(old_labels_dict['image']):
            if image == name:
                label = old_labels_dict['labels'][j]
                aisc_labels_array[0] = image
                aisc_labels_array[label] = 1

new_labels_array = np.vstack(isic_labels_array, aisc_labels_array)

full_labels = pd.fromarray(new_labels_array)

full_labels.to_csv(new_labels_path)


