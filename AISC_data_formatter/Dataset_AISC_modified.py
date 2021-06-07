#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:09:03 2020

@author: mohan
"""
import os, sys, random, pickle, collections
import json
import glob
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision
import albumentations as A
import AISC_2_ISIC
import pandas as pd
import shutil
class AISCDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_path='~/scratch/melanoma/AISC/', new_ext='bmp'):
        assert mode in ['train', 'val', 'all']
        # pickle.dumps(transform)
        data_path = os.path.expanduser(data_path)
        self.data_path = data_path
        self.folder = 'images'
        
        with open(os.path.join(data_path, 'images.json'), 'r') as f:
            json_file = json.load(f)
        self.images_dicts = json_file['images']
        
        #subfolders = ['photodermoscopybaseline', 'photodermoscopybaseline-cropped',
        #              'photodermoscopyfollowup', 'photodermoscopyfollowup-cropped']
        #for sub_f in subfolders:
        #    Helpers.convert_to_new_ext(os.path.join(data_path, self.folder, sub_f), new_ext)
        data_path_ext = self.data_path.replace('Odense_Data', 'Odense_Data_Preprocessed' + new_ext)
        #data_path_ext = der hvor images ligger med præprocessering
        
        self.labels = {}
        self.im_uid_dict = {}
        
        self.lesions = collections.defaultdict(lambda: [])
        self.mcq_lesion_uids = []
        peter = 0
        for im_dict in self.images_dicts:
            lesion_uid = im_dict['lesion']['uid']
            if 'uncroppedPath' in im_dict:

                im_path = im_dict['uncroppedPath']
            else:
                im_path= im_dict['croppedPath']

        
            # self.im_uid_dict[os.path.split(im_path)[1][:-5]] = im_dict
            self.im_uid_dict[im_dict['uid']] = im_dict
            im_path = im_path.replace('images', r'C:\Users\ptrkm\Bachelor\Odense_Data_Preprocessed')

            im_path = os.path.join(data_path_ext, im_path)

            if im_path.endswith('octet-stream'):
                print(im_path)
                print('Lesion UID', lesion_uid)
                continue
            if im_dict['lesion']['mcqOnly']:
                self.mcq_lesion_uids.append(lesion_uid)
                if mode!='all':
                    continue

            assert(os.path.exists(im_path))
            self.lesions[lesion_uid].append((im_path, im_dict))

        lesion_uids = sorted(list(self.lesions.keys()))
        if mode!='all':
            assert all([l not in lesion_uids for l in self.mcq_lesion_uids])
        random.seed(0)
        random.shuffle(lesion_uids)
        
        #Move duplicated lesions to the training set
        #These are actually lesion uids, which makes more sense
        with open('Duplicate_lesion_uids.txt', 'r') as f:
            lines = f.readlines()
        duplicates = [l.split()[1] for l in lines] + [l.split()[3] for l in lines]
        for idx1, dup in enumerate(duplicates):
            idx2 = lesion_uids.index(dup)
            lesion_uids[idx1], lesion_uids[idx2] = lesion_uids[idx2], lesion_uids[idx1]
        self.duplicate_image_uids = duplicates
        
        split_point = int(len(lesion_uids)*.4)
        lesion_uids = {'train': lesion_uids[:split_point], 
                       'val':   lesion_uids[split_point:],
                       'all':   lesion_uids[:]}[mode]
        
        self.images = []
        discarded_lesions = 0
        for lesion_uid in lesion_uids:
            lesion = sorted(self.lesions[lesion_uid])
            assert len(set([d['lesion']['diagnosis'] for p, d in lesion]))==1
            assert len(set([d['lesion']['patient']['uid'] for p, d in lesion]))==1
            im_dict = lesion[-1][1]
            diag = im_dict['lesion']['diagnosis']
            if diag in AISC_2_ISIC.diag_dict:
                diag_idx = AISC_2_ISIC.diag_dict[diag]
            else:
                raise ValueError('Diagnoisis ', diag, 'is unknown.')
            if diag_idx>=0:
                ims = {'train': lesion,
                       'all':   lesion,
                       'val':   [lesion[-1]]}[mode]
                for im_path, im_dict in ims:
                    im_name = os.path.splitext(os.path.split(im_path)[1])[0]
                    self.images.append(im_path)
                    self.labels[im_name] = diag_idx
            else:
                discarded_lesions += 1
        print("Discarded lesions", discarded_lesions)

        assert(len(self)>100)
       
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx,new_path):
        image_path = self.images[idx]
        image_path = image_path.replace('images', self.folder)
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]
        image_name2 = os.path.split(image_path)[1]
        # image = cv2.imread(image_path)
        new_path = os.path.join(new_path, image_name2)
        # shutil.copyfile(image_path,new_path)

        label = self.labels[image_name]

        return label, image_name
all = AISCDataset(mode='all',data_path =r'C:\Users\ptrkm\Bachelor\Odense_Data',new_ext='')
print(len(all.images))
breakpoint()

validation= AISCDataset(mode = 'val',data_path=r'C:\Users\ptrkm\Bachelor\Odense_Data',new_ext='')
training = AISCDataset(mode = 'train',data_path=r'C:\Users\ptrkm\Bachelor\Odense_Data',new_ext='')
isic_label_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_frame = np.zeros((len(validation.images),len(isic_label_names)))
save_path = r'C:\Users\ptrkm\Bachelor\DataFinal'
imagenames = []
counter = 0
for i in range(len(validation.images)):
    label, image_name = validation.__getitem__(i,save_path)
    if label != 7:
        labels_frame[counter,label] = 1
        counter += 1
        imagenames.append(image_name)

labels_frame_train = np.zeros((len(training.images),len(isic_label_names)))
image_names_train = []
counter = 0
for idx, image in enumerate(training.images):
    label, image_name = training.__getitem__(idx,save_path)
    if label != 7:
        labels_frame_train[counter,label] = 1
        counter += 1
        image_names_train.append(image_name)

labels_frame = labels_frame[:,:-1]

labels_frame = labels_frame[:len(imagenames),:]

labels_frame_train = labels_frame_train[:,:-1]
labels_frame_train = labels_frame_train[:len(image_names_train),:]

# final_pd = pd.DataFrame(data=labels_frame, columns = isic_label_names)
# final_pd.insert(loc = 0, column = 'image',value =imagenames)
# final_pd.to_csv(r'C:\Users\ptrkm\Bachelor\labels_2018.csv',sep = ',', index = False)

num_each = np.floor(np.array([179.03507611, 950.66578425,  97.36995367,45.02051621,227.19655857,  46.06750496,  36.64460622]))
new_image_list = []
indices_list = []
for i in range(labels_frame.shape[1]):
    indices = np.where(labels_frame[:,i] == 1)[0]

    idx = np.random.choice(indices,int(num_each[i]),replace=False)

    boolan_array = np.array([1 if j in idx else 0 for j,i in enumerate(imagenames)])
    boolan_array_train = np.array([1 if j not in idx and j in indices else 0 for j,i in enumerate(imagenames)])
    indices_list += idx.tolist()

    new_image_list += (np.array(imagenames)[boolan_array==1]).tolist()
    image_names_train += (np.array(imagenames)[boolan_array_train==1]).tolist()


labels_aisc_isic = pd.read_csv(r'C:\Users\ptrkm\Bachelor\labels_aisc_isic.csv')

pcl = {}
pcl['trainIndCV'] = []
pcl['valIndCV'] = []

labels_list = labels_aisc_isic['image']
no_go_list = labels_aisc_isic['SCC']
for j, image in enumerate(labels_list):
    if image in new_image_list:
        pcl['valIndCV'].append(j)
    elif 'ISIC' in image or image in image_names_train:
        if no_go_list[j] != 1:
            pcl['trainIndCV'].append(j)
    if j % 1000 ==0:
        print(j)

pcl['trainIndCV'] = np.array(pcl['trainIndCV'])
pcl['valIndCV'] = np.array(pcl['valIndCV'])

with open(r'C:\Users\ptrkm\Bachelor\indices_aisc_plus_isic.pkl','wb') as handle:
    pickle.dump(pcl,handle, protocol=pickle.HIGHEST_PROTOCOL)







breakpoint()











breakpoint()