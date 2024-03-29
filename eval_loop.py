import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import models
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
import csv
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import imagesize

# add configuration file
# Dictionary for model configuration

if sys.argv[2] == 'original_rr':
    network_list_eff = ['efficientnet_b0_rr', 'efficientnet_b1_rr', 'efficientnet_b2_rr',
                        'efficientnet_b3_rr', 'efficientnet_b4_rr', 'efficientnet_b5_rr', 'res101_rr',
                        'se_resnet101_rr', 'nasnetamobile_rr', 'resnext101_32_8_wsl_rr']
    network_list_eff = ['2019_rr.test_' + i for i in network_list_eff]

elif sys.argv[2] == 'original_ss':
    network_list_eff = ['efficientnet_b0_ss_autoaugment', 'efficientnet_b1_ss', 'efficientnet_b2_ss',
                        'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss', 'efficientnet_b6_ss',
                        'senet154_ss']
    network_list_eff = ['2019.test_' + i for i in network_list_eff]

elif sys.argv[2] == '2019_daisy_rr':
    network_list_eff = ['efficientnet_b0_rr','efficientnet_b1_rr','efficientnet_b2_rr',
                    'efficientnet_b3_rr', 'efficientnet_b4_rr','efficientnet_b5_rr' ]
    network_list_eff = ['2019_daisy.' + i for i in network_list_eff]

elif sys.argv[2] == '2019_daisy_ss':
    network_list_eff = ['2019_daisy.senet154_ss']

elif sys.argv[2] == 'wb1_late_cc':
    network_list_eff = ['2019_rr.resnet101_rr_wb1_cc_late']

elif sys.argv[2] == 'wb2':
    network_list_eff = ['2019_rr.resnet101_rr_wb2_no_cc']

elif sys.argv[2] == '2018_mixed_aisc_val':
    network_list_eff = ['efficientnet_b0_rr', 'efficientnet_b1_rr', 'efficientnet_b2_rr',
                        'efficientnet_b3_rr', 'efficientnet_b4_rr', 'efficientnet_b5_rr',
                        'resnet101_rr','se_resnet101_rr', 'nasnetamobile_rr', 'resnext101_32_8_rr','efficientnet_b0_ss',
                        'efficientnet_b1_ss', 'efficientnet_b2_ss',
                        'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss', 'efficientnet_b6_ss',
                        'senet154_ss']
    network_list_eff = ['2018_mixed.' + i for i in network_list_eff]

elif sys.argv[2] == '2019_daisy_on_2018':
    network_list_eff = ['efficientnet_b0_rr', 'efficientnet_b1_rr', 'efficientnet_b2_rr',
                        'efficientnet_b3_rr', 'efficientnet_b4_rr', 'efficientnet_b5_rr','efficientnet_b0_ss',
                        'efficientnet_b1_ss', 'efficientnet_b2_ss',
                        'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss', 'efficientnet_b6_ss',
                        'senet154_ss']
    network_list_eff = ['2019_daisy.' + i for i in network_list_eff]

elif sys.argv[2] == 'original_ss':
    network_list_eff = ['efficientnet_b0_ss_autoaugment', 'efficientnet_b1_ss', 'efficientnet_b2_ss',
                        'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss', 'efficientnet_b6_ss',
                        'senet154_ss']
    network_list_eff = ['2019.test_' + i for i in network_list_eff]

elif sys.argv[2] == '2019daisy_effb6':
    network_list_eff = ['2019_daisy.efficientnet_b6_ss']

elif sys.argv[2] == 'original_effb6':
    network_list_eff = ['2019.test_efficientnet_b6_ss']

elif sys.argv[2] == '2019effs':
    network_list_eff = ['efficientnet_b3_ss', 'efficientnet_b4_rr',
                         'efficientnet_b4_ss', 'efficientnet_b5_rr_CVSet0', 'efficientnet_b5_rr_CVSet1', 'efficientnet_b5_rr_CVSet2',
                        'efficientnet_b5_ss_CVSet0', 'efficientnet_b5_ss_CVSet1', 'efficientnet_b5_ss_CVSet2',
                        'efficientnet_b6_ss_CVSet0', 'efficientnet_b6_ss_CVSet1', 'efficientnet_b6_ss_CVSet2']
    network_list_eff = ['2019_effs_daisy.' + i for i in network_list_eff]

elif sys.argv[2] == '2019_aisc_full':
    network_list_eff = ['efficientnet_b3_ss', 'efficientnet_b4_rr',
                         'efficientnet_b4_ss', 'efficientnet_b4_rr','efficientnet_b5_ss',
                         'efficientnet_b5_rr','efficientnet_b6_ss']
    network_list_eff = ['2019_aisc_full.' + i for i in network_list_eff]

elif sys.argv[2] == '2019_aisc_full_b3_rr':
    network_list_eff = ['2019_aisc_full.efficientnet_b3_rr']

network_list = network_list_eff
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.' + sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)
mdlParams['color_augmentation'] = False
for networks_peter in network_list:
    print(networks_peter)
    if 'rr' in networks_peter:
        crop_strategy = 'multideterm1sc4f4'
    else:
        crop_strategy = 'multiorder36'
        print("Crop strategy: " + str(crop_strategy))
    # If there is another argument, its which checkpoint should be used
    if len(sys.argv) > 6:
        if 'last' in sys.argv[6]:
            mdlParams['ckpt_name'] = 'checkpoint-'
        else:
            mdlParams['ckpt_name'] = 'checkpoint_best-'
        if 'first' in sys.argv[6]:
            mdlParams['use_first'] = True
    else:
        mdlParams['ckpt_name'] = 'checkpoint-'

    # Set visible devices
    mdlParams['numGPUs'] = [[int(s) for s in re.findall(r'\d+', sys.argv[6])][-1]]
    cuda_str = ""
    for i in range(len(mdlParams['numGPUs'])):
        cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
        if i is not len(mdlParams['numGPUs']) - 1:
            cuda_str = cuda_str + ","
    print("Devices to use:", cuda_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

    # If there is another argument, also use a meta learner
    if len(sys.argv) > 7:
        if 'HAMONLY' in sys.argv[7]:
            mdlParams['eval_on_ham_only'] = True

        # Import model config
    model_cfg = importlib.import_module('cfgs.' + networks_peter)
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    # Path name where model is saved is the fourth argument
    if 'NONE' in sys.argv[5]:
        mdlParams['saveDirBase'] = mdlParams['saveDir'] + networks_peter
    else:
        mdlParams['saveDirBase'] = sys.argv[5]

    # Third is multi crop yes no
    if 'multi' in crop_strategy:
        if 'rand' in crop_strategy:
            mdlParams['numRandValSeq'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][0]
            print("Random sequence number", mdlParams['numRandValSeq'])
        else:
            mdlParams['numRandValSeq'] = 0
        mdlParams['multiCropEval'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][-1]
        mdlParams['voting_scheme'] = sys.argv[4]
        if 'scale' in crop_strategy:
            print("Multi Crop and Scale Eval with crop number:", mdlParams['multiCropEval'], " Voting scheme: ",
                  mdlParams['voting_scheme'])
            mdlParams['orderedCrop'] = False
            mdlParams['scale_min'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][-2] / 100.0
        elif 'determ' in crop_strategy:
            # Example application: multideterm5sc3f2
            mdlParams['deterministic_eval'] = True
            mdlParams['numCropPositions'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][-3]
            num_scales = [int(s) for s in re.findall(r'\d+', crop_strategy)][-2]
            all_scales = [1.0, 0.5, 0.75, 0.25, 0.9, 0.6, 0.4]
            mdlParams['cropScales'] = all_scales[:num_scales]
            mdlParams['cropFlipping'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][-1]
            print("deterministic eval with crops number", mdlParams['numCropPositions'], "scales", mdlParams['cropScales'],
                  "flipping", mdlParams['cropFlipping'])
            mdlParams['multiCropEval'] = mdlParams['numCropPositions'] * len(mdlParams['cropScales']) * mdlParams[
                'cropFlipping']
            mdlParams['offset_crop'] = 0.2
        elif 'order' in crop_strategy:
            mdlParams['orderedCrop'] = True
            mdlParams['var_im_size'] = True
            if mdlParams.get('var_im_size', True):
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
                    if mdlParams.get('resize_large_ones') is not None:
                        if width == mdlParams['large_size'] and height == mdlParams['large_size']:
                            width, height = (mdlParams['resize_large_ones'], mdlParams['resize_large_ones'])
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
                    if mdlParams.get('resize_large_ones') is not None:
                        if width_test == mdlParams['large_size'] and height_test == mdlParams['large_size']:
                            width_test, height_test = (mdlParams['resize_large_ones'], mdlParams['resize_large_ones'])
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
            else:
                # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
                mdlParams['cropPositions'] = np.zeros([mdlParams['multiCropEval'], 2], dtype=np.int64)
                if mdlParams['multiCropEval'] == 5:
                    numCrops = 4
                elif mdlParams['multiCropEval'] == 7:
                    numCrops = 9
                    mdlParams['cropPositions'] = np.zeros([9, 2], dtype=np.int64)
                else:
                    numCrops = mdlParams['multiCropEval']
                ind = 0
                for i in range(np.int32(np.sqrt(numCrops))):
                    for j in range(np.int32(np.sqrt(numCrops))):
                        mdlParams['cropPositions'][ind, 0] = mdlParams['input_size'][0] / 2 + i * (
                                    (mdlParams['input_size_load'][0] - mdlParams['input_size'][0]) / (
                                        np.sqrt(numCrops) - 1))
                        mdlParams['cropPositions'][ind, 1] = mdlParams['input_size'][1] / 2 + j * (
                                    (mdlParams['input_size_load'][1] - mdlParams['input_size'][1]) / (
                                        np.sqrt(numCrops) - 1))
                        ind += 1
                # Add center crop
                if mdlParams['multiCropEval'] == 5:
                    mdlParams['cropPositions'][4, 0] = mdlParams['input_size_load'][0] / 2
                    mdlParams['cropPositions'][4, 1] = mdlParams['input_size_load'][1] / 2
                if mdlParams['multiCropEval'] == 7:
                    mdlParams['cropPositions'] = np.delete(mdlParams['cropPositions'], [3, 7], 0)
                    # Sanity checks
                print("Positions val", mdlParams['cropPositions'])
                # Test image sizes
                test_im = np.zeros(mdlParams['input_size_load'])
                height = mdlParams['input_size'][0]
                width = mdlParams['input_size'][1]
                for i in range(mdlParams['multiCropEval']):
                    im_crop = test_im[np.int32(mdlParams['cropPositions'][i, 0] - height / 2):np.int32(
                        mdlParams['cropPositions'][i, 0] - height / 2) + height,
                              np.int32(mdlParams['cropPositions'][i, 1] - width / 2):np.int32(
                                  mdlParams['cropPositions'][i, 1] - width / 2) + width, :]
                    print("Shape", i + 1, im_crop.shape)
            print("Multi Crop with order with crop number:", mdlParams['multiCropEval'], " Voting scheme: ",
                  mdlParams['voting_scheme'])
            if 'flip' in crop_strategy:
                # additional flipping, example: flip2multiorder16
                mdlParams['eval_flipping'] = [int(s) for s in re.findall(r'\d+', crop_strategy)][-2]
                print("Additional flipping", mdlParams['eval_flipping'])
        else:
            print("Multi Crop Eval with crop number:", mdlParams['multiCropEval'], " Voting scheme: ",
                  mdlParams['voting_scheme'])
            mdlParams['orderedCrop'] = False
    else:
        mdlParams['multiCropEval'] = 0
        mdlParams['orderedCrop'] = False

    # Set training set to eval mode
    mdlParams['trainSetState'] = 'eval'

    if mdlParams['numClasses'] == 9 and mdlParams.get('no_c9_eval', False):
        num_classes = mdlParams['numClasses'] - 1
    else:
        num_classes = mdlParams['numClasses']
    # Save results in here
    allData = {}
    allData['f1Best'] = np.zeros([mdlParams['numCV']])
    allData['sensBest'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['specBest'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['accBest'] = np.zeros([mdlParams['numCV']])
    allData['waccBest'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['aucBest'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['convergeTime'] = {}
    allData['bestPred'] = {}
    allData['bestPredMC'] = {}
    allData['targets'] = {}
    allData['extPred'] = {}
    allData['f1Best_meta'] = np.zeros([mdlParams['numCV']])
    allData['sensBest_meta'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['specBest_meta'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['accBest_meta'] = np.zeros([mdlParams['numCV']])
    allData['waccBest_meta'] = np.zeros([mdlParams['numCV'], num_classes])
    allData['aucBest_meta'] = np.zeros([mdlParams['numCV'], num_classes])
    # allData['convergeTime'] = {}
    allData['bestPred_meta'] = {}
    allData['targets_meta'] = {}
    allData['all_images'] = {}

    if len(sys.argv) > 8:
        for cv in range(mdlParams['numCV']):
            # Reset model graph
            importlib.reload(models)
            # importlib.reload(torchvision)
            # Collect model variables
            modelVars = {}
            modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # define new folder, take care that there might be no labels
            print("Creating predictions for path ", sys.argv[8])
            path1 = sys.argv[8]
            # All files in that set
            files = sorted(glob(path1 + '/*'))
            # Define new paths
            mdlParams['im_paths'] = []
            mdlParams['meta_list'] = []
            for j in range(len(files)):
                inds = [int(s) for s in re.findall(r'\d+', files[j])]
                mdlParams['im_paths'].append(files[j])
                if 'ISIC_' in files[j]:
                    if mdlParams.get('meta_features', None) is not None:
                        for key in mdlParams['meta_dict']:
                            if key in files[j]:
                                mdlParams['meta_list'].append(mdlParams['meta_dict'][key])
            if mdlParams.get('meta_features', None) is not None:
                # Meta data
                mdlParams['meta_array'] = np.array(mdlParams['meta_list'])
                # Add empty labels
            mdlParams['labels_array'] = np.zeros([len(mdlParams['im_paths']), mdlParams['numClasses']], dtype=np.float32)
            # Define everything as a valind set
            mdlParams['valInd'] = np.array(np.arange(len(mdlParams['im_paths'])))
            mdlParams['trainInd'] = mdlParams['valInd']
            if mdlParams.get('var_im_size', False):
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
                    if mdlParams.get('resize_large_ones') is not None:
                        if width == mdlParams['large_size'] and height == mdlParams['large_size']:
                            width, height = (mdlParams['resize_large_ones'], mdlParams['resize_large_ones'])
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
                # test_im = np.zeros(mdlParams['input_size_load'])
                test_im = np.zeros(mdlParams['input_size'])
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
                    if mdlParams.get('resize_large_ones') is not None:
                        if width_test == mdlParams['large_size'] and height_test == mdlParams['large_size']:
                            width_test, height_test = (mdlParams['resize_large_ones'], mdlParams['resize_large_ones'])
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
            mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)

            if mdlParams['balance_classes'] == 9:
                # Only use HAM indicies for calculation
                print("Balance 9")
                indices_ham = mdlParams['trainInd']
                if mdlParams['numClasses'] == 9:
                    # Class weights for the AISC dataset
                    class_weights_ = np.array(
                        [0.06544445, 0.81089767, 0.02461471, 0.00803448, 0.06332627, 0.008984, 0.01599591, 0.00270251])
                    # print("class before",class_weights_)
                    class_weights = np.zeros([mdlParams['numClasses']])
                    class_weights[:8] = class_weights_
                elif mdlParams['numClasses'] == 8:
                    # Class weights for the AISC dataset
                    class_weights = np.array(
                        [0.06544445, 0.81089767, 0.02461471, 0.00803448, 0.06332627, 0.008984, 0.01599591, 0.00270251])
                elif mdlParams['numClasses'] == 7:
                    class_weights = np.array(
                        [0.06544445, 0.81089767, 0.02461471, 0.00803448, 0.06332627, 0.008984, 0.01599591]
                    )
                print("Current class weights", class_weights)
                if isinstance(mdlParams['extra_fac'], float):
                    class_weights = np.power(class_weights, mdlParams['extra_fac'])
                else:
                    class_weights = class_weights * mdlParams['extra_fac']
                print("Current class weights with extra", class_weights)

                # Set up dataloaders

            # Meta scaler
            if mdlParams.get('meta_features', None) is not None and mdlParams['scale_features']:
                mdlParams['feature_scaler_meta'] = sklearn.preprocessing.StandardScaler().fit(
                    mdlParams['meta_array'][mdlParams['trainInd'], :])
                # print("scaler mean",mdlParams['feature_scaler_meta'].mean_,"var",mdlParams['feature_scaler_meta'].var_)
            # For train
            # dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
            # For val
            dataset_val = utils.ISICDataset(mdlParams, 'valInd')
            if mdlParams['multiCropEval'] > 0:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'],
                                                            shuffle=False, num_workers=8, pin_memory=True)
            else:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False,
                                                            num_workers=8, pin_memory=True)
            # modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True,
            #                                               num_workers=8, pin_memory=True)

            # Define model
            modelVars['model'] = models.getModel(mdlParams)()
            if 'Dense' in mdlParams['model_type']:
                if mdlParams['input_size'][0] != 224:
                    modelVars['model'] = utils.modify_densenet_avg_pool(modelVars['model'])
                    # print(modelVars['model'])
                num_ftrs = modelVars['model'].classifier.in_features
                modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
                # print(modelVars['model'])
            elif 'dpn' in mdlParams['model_type']:
                num_ftrs = modelVars['model'].classifier.in_channels
                modelVars['model'].classifier = nn.Conv2d(num_ftrs, mdlParams['numClasses'], [1, 1])
                # modelVars['model'].add_module('real_classifier',nn.Linear(num_ftrs, mdlParams['numClasses']))
                # print(modelVars['model'])
            elif 'efficient' in mdlParams['model_type'] and (
                    '0' in mdlParams['model_type'] or '1' in mdlParams['model_type'] \
                    or '2' in mdlParams['model_type'] or '3' in mdlParams['model_type']):
                num_ftrs = modelVars['model'].classifier.in_features
                modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])

            elif 'efficient' in mdlParams['model_type']:
                num_ftrs = modelVars['model']._fc.in_features
                modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])

            elif 'wsl' in mdlParams['model_type'] or 'Resnet' in mdlParams['model_type'] or 'Inception' in mdlParams[
                'model_type']:
                # Do nothing, output is prepared
                num_ftrs = modelVars['model'].fc.in_features
                modelVars['model'].fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
            else:
                num_ftrs = modelVars['model'].last_linear.in_features
                modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])
                # Take care of meta case
            if mdlParams.get('meta_features', None) is not None:
                modelVars['model'] = models.modify_meta(mdlParams, modelVars['model'])
            modelVars['model'] = modelVars['model'].to(modelVars['device'])
            # summary(modelVars['model'], (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
            # Loss, with class weighting
            # Loss, with class weighting
            if mdlParams['balance_classes'] == 9:
                modelVars['criterion'] = nn.CrossEntropyLoss(
                    weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
            # Observe that all parameters are being optimized
            modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

            # Decay LR by a factor of 0.1 every 7 epochs
            modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'],
                                                         gamma=1 / np.float32(mdlParams['LRstep']))

            # Define softmax
            modelVars['softmax'] = nn.Softmax(dim=1)

            # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
            files = glob(mdlParams['saveDir'] + '/*')
            global_steps = np.zeros([len(files)])
            for i in range(len(files)):
                # Use meta files to find the highest index
                if 'checkpoint' not in files[i]:
                    continue
                if mdlParams['ckpt_name'] not in files[i]:
                    continue
                # Extract global step
                nums = [int(s) for s in re.findall(r'\d+', files[i])]
                global_steps[i] = nums[-1]
            # Create path with maximum global step found, if first is not wanted
            global_steps = np.sort(global_steps)
            if mdlParams.get('use_first') is not None:
                chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(global_steps[-2])) + '.pt'
            else:
                chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps))) + '.pt'
            print("Restoring: ", chkPath)

            # Load
            state = torch.load(chkPath)
            # Initialize model and optimizer
            modelVars['model'].load_state_dict(state['state_dict'])
            # modelVars['optimizer'].load_state_dict(state['optimizer'])
            # Get predictions or learn on pred
            modelVars['model'].eval()
            # Get predictions
            # Turn off the skipping of the last class
            mdlParams['no_c9_eval'] = False
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc,images = utils.getErrClassification_mgpu(
                mdlParams, 'valInd', modelVars)
            # Save predictions
            allData['extPred'][cv] = predictions
            allData['all_images'][cv] = images
            print("extPred shape", allData['extPred'][cv].shape)
            pklFileName = networks_peter + "_" + sys.argv[6] + "_" + str(int(np.max(global_steps))) + "_predn.pkl"

    # Mean results over all folds
    np.set_printoptions(precision=4)
    print("-------------------------------------------------")
    print("Mean over all Folds")
    print("-------------------------------------------------")
    print("F1 Score", np.array([np.mean(allData['f1Best'])]), "+-", np.array([np.std(allData['f1Best'])]))
    print("Sensitivtiy", np.mean(allData['sensBest'], 0), "+-", np.std(allData['sensBest'], 0))
    print("Specificity", np.mean(allData['specBest'], 0), "+-", np.std(allData['specBest'], 0))
    print("Mean Specificity", np.array([np.mean(allData['specBest'])]), "+-",
          np.array([np.std(np.mean(allData['specBest'], 1))]))
    print("Accuracy", np.array([np.mean(allData['accBest'])]), "+-", np.array([np.std(allData['accBest'])]))
    print("Per Class Accuracy", np.mean(allData['waccBest'], 0), "+-", np.std(allData['waccBest'], 0))
    print("Weighted Accuracy", np.array([np.mean(allData['waccBest'])]), "+-",
          np.array([np.std(np.mean(allData['waccBest'], 1))]))
    print("AUC", np.mean(allData['aucBest'], 0), "+-", np.std(allData['aucBest'], 0))
    print("Mean AUC", np.array([np.mean(allData['aucBest'])]), "+-", np.array([np.std(np.mean(allData['aucBest'], 1))]))
    print("sum check#######", np.sum(predictions))
    # Save dict with results
    mdlParams['saveDirBase'] = sys.argv[3]
    with open(mdlParams['saveDirBase'] + "/" + pklFileName, 'wb') as f:
        pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)