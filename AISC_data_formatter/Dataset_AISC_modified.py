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

class AISCDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_path='~/scratch/melanoma/AISC/', new_ext='bmp'):
        assert mode in ['train', 'val', 'all']
        pickle.dumps(transform)
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
        #data_path_ext = self.data_path.replace('melanoma', 'melanoma_' + new_ext)
        #data_path_ext = der hvor images ligger med præprocessering
        
        self.labels = {}
        self.im_uid_dict = {}
        
        self.lesions = collections.defaultdict(lambda: [])
        self.mcq_lesion_uids = []
        for im_dict in self.images_dicts:
            lesion_uid = im_dict['lesion']['uid']
            if 'uncroppedPath' in im_dict:
                im_path = im_dict['uncroppedPath']
            else:
                im_path= im_dict['croppedPath']
        
            self.im_uid_dict[os.path.split(im_path)[1][:-5]] = im_dict
            
            im_path = os.path.join(data_path_ext, im_path.replace('jpeg', new_ext))
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
        
        split_point = int(len(lesion_uids)*.8)
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
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_path = image_path.replace('images', self.folder)
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]

        image = cv2.imread(image_path)
        
        label = self.labels[image_name]
        return image, label
