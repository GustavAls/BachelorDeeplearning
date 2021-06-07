import os
import sys
import h5py
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle
import imagesize

def init(mdlParams_):
    mdlParams = {}
    local_path = '/isic2019/'
    mdlParams['saveDir'] = mdlParams_['pathBase'] + '/'
    # Data is loaded from here
    mdlParams['dataDir'] = mdlParams_['pathBase'] + local_path

    ### Model Selection ###
    mdlParams['model_type'] = 'efficientnet-b5'
    mdlParams['dataset_names'] = ['official']  # ,'sevenpoint_rez3_ll']
    mdlParams['file_ending'] = '.jpg'
    mdlParams['exclude_inds'] = False
    mdlParams['same_sized_crops'] = False
    mdlParams['multiCropEval'] = 9
    mdlParams['var_im_size'] = False
    mdlParams['orderedCrop'] = False
    mdlParams['voting_scheme'] = 'average'
    mdlParams['classification'] = True
    mdlParams['balance_classes'] = 9
    mdlParams['extra_fac'] = 1.0
    mdlParams['numClasses'] = 8
    mdlParams['no_c9_eval'] = True
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 1
    mdlParams['trans_norm_first'] = True
    # Scale up for b1-b7
    mdlParams['input_size'] = [456, 456, 3]

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 6  # *len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = 0.000015  # *len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 25
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 50
    # Divide learning rate by this value
    mdlParams['LRstep'] = 5
    # Maximum number of training iterations
    mdlParams['training_steps'] = 60  # 250
    # Display error every X steps
    mdlParams['display_step'] = 10
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])
    mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])

    # Data AUG
    # mdlParams['full_color_distort'] = True
    mdlParams['autoaugment'] = False
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['shear'] = 10
    mdlParams['cutout'] = 16
    ### Data ###
    mdlParams['preload'] = False
    # Labels first
    # Targets, as dictionary, indexed by im file name
    mdlParams['labels_dict'] = {}
    path1 = mdlParams['dataDir'] + '/AISC_ISIC_full_labels/'
    # path1 = mdlParams['dataDir'] + '\labels\\'
    # All sets
    allSets = glob(path1 + '*/')
    # allSets = glob(path1 + '*\\')
    # Go through all sets
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue
        # Find csv file
        files = sorted(glob(allSets[i] + '*'))
        for j in range(len(files)):
            if 'csv' in files[j]:
                break
        # Load csv file
        with open(files[j], newline='') as csvfile:
            labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in labels_str:
                if 'image' == row[0]:
                    continue
                # if 'ISIC' in row[0] and '_downsampled' in row[0]:
                #    print(row[0])
                if row[0] + '_downsampled' in mdlParams['labels_dict']:
                    print("removed", row[0] + '_downsampled')
                    continue
                if mdlParams['numClasses'] == 7:
                    mdlParams['labels_dict'][row[0]] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7]))])
                elif mdlParams['numClasses'] == 8:
                    if len(row) < 9 or row[8] == '':
                        class_8 = 0
                    else:
                        class_8 = int(float(row[8]))
                    mdlParams['labels_dict'][row[0]] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7])), class_8])
                elif mdlParams['numClasses'] == 9:
                    if len(row) < 9 or row[8] == '':
                        class_8 = 0
                    else:
                        class_8 = int(float(row[8]))
                    if len(row) < 10 or row[9] == '':
                        class_9 = 0
                    else:
                        class_9 = int(float(row[9]))
                    mdlParams['labels_dict'][row[0]] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7])), class_8, class_9])
    # Save all im paths here
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    # Define the sets
    path1 = mdlParams['dataDir'] + '/AISC_ISIC_full/official/'
    # All sets
    allSets = sorted(glob(path1 + '*/'))
    # allSets = sorted(glob(path1 + '*\\'))

    # Ids which name the folders
    # Make official first dataset
    for i in range(len(allSets)):
        if mdlParams['dataset_names'][0] in allSets[i]:
            temp = allSets[i]
            allSets.remove(allSets[i])
            allSets.insert(0, temp)
    print(allSets)
    # Set of keys, for marking old HAM10000
    mdlParams['key_list'] = []
    if mdlParams['exclude_inds']:
        with open(mdlParams['saveDir'] + 'indices_exclude.pkl', 'rb') as f:
            indices_exclude = pickle.load(f)
        exclude_list = []
    for i in range(len(allSets)):
        # All files in that set
        files = sorted(glob(allSets[i] + '*'))
        # Check if there is something in there, if not, discard
        if len(files) == 0:
            continue
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue
        for key in mdlParams['labels_dict']:
            for j in range(len(files)):
                name = files[j].split('/')[-1]
                name = name.split('.')[0]
                if key == name:
                    mdlParams['labels_list'].append(mdlParams['labels_dict'][key])
                    mdlParams['im_paths'].append(files[j])
                    break

    # Convert label list to array
    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])

    ### Define Indices ###
    with open(mdlParams['saveDir'] + 'indices_aisc_isic_2019.pkl', 'rb') as f:
        indices = pickle.load(f)

    mdlParams['trainIndCV'] = indices['trainIndCV']
    mdlParams['valIndCV'] = indices['valIndCV']
    val_labels = mdlParams['labels_array'][mdlParams['valIndCV']]
    print("Class distribution in b3 rr configuration")
    print(np.sum(val_labels, axis=0))

    print("ValIndCV:")
    print(mdlParams['valIndCV'])

    print("Train")
    # Use this for ordered multi crops
    if mdlParams['orderedCrop']:
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        mdlParams['cropPositions'] = np.zeros([len(mdlParams['im_paths']), mdlParams['multiCropEval'], 2],
                                              dtype=np.int64)
        # mdlParams['imSizes'] = np.zeros([len(mdlParams['im_paths']),mdlParams['multiCropEval'],2],dtype=np.int64)
        for u in range(len(mdlParams['im_paths'])):
            height, width = imagesize.get(mdlParams['im_paths'][u])
            if width < mdlParams['input_size'][0]:
                height = int(mdlParams['input_size'][0] / float(width)) * height
                width = mdlParams['input_size'][0]
            if height < mdlParams['input_size'][0]:
                width = int(mdlParams['input_size'][0] / float(height)) * width
                height = mdlParams['input_size'][0]
            ind = 0
            for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                    mdlParams['cropPositions'][u, ind, 0] = mdlParams['input_size'][0] / 2 + i * (
                            (width - mdlParams['input_size'][1]) / (np.sqrt(mdlParams['multiCropEval']) - 1))
                    mdlParams['cropPositions'][u, ind, 1] = mdlParams['input_size'][1] / 2 + j * (
                            (height - mdlParams['input_size'][0]) / (np.sqrt(mdlParams['multiCropEval']) - 1))
                    # mdlParams['imSizes'][u,ind,0] = curr_im_size[0]

                    ind += 1
        # Sanity checks
        # print("Positions",mdlParams['cropPositions'])
        # Test image sizes
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for u in range(len(mdlParams['im_paths'])):
            height_test, width_test = imagesize.get(mdlParams['im_paths'][u])
            if width_test < mdlParams['input_size'][0]:
                height_test = int(mdlParams['input_size'][0] / float(width_test)) * height_test
                width_test = mdlParams['input_size'][0]
            if height_test < mdlParams['input_size'][0]:
                width_test = int(mdlParams['input_size'][0] / float(height_test)) * width_test
                height_test = mdlParams['input_size'][0]
            test_im = np.zeros([width_test, height_test])
            for i in range(mdlParams['multiCropEval']):
                im_crop = test_im[np.int32(mdlParams['cropPositions'][u, i, 0] - height / 2):np.int32(
                    mdlParams['cropPositions'][u, i, 0] - height / 2) + height,
                          np.int32(mdlParams['cropPositions'][u, i, 1] - width / 2):np.int32(
                              mdlParams['cropPositions'][u, i, 1] - width / 2) + width]
                if im_crop.shape[0] != mdlParams['input_size'][0]:
                    print("Wrong shape", im_crop.shape[0], mdlParams['im_paths'][u])
                if im_crop.shape[1] != mdlParams['input_size'][1]:
                    print("Wrong shape", im_crop.shape[1], mdlParams['im_paths'][u])
    return mdlParams