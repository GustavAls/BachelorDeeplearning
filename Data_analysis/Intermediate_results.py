

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
import seaborn as sns
show_training_results = False
show_AISC_results = True
path = r'C:\Users\ptrkm\Bachelor\AISC_results'
average_voting = True
majority_voting = False
our_ensemble = ['res_101_rr', 'efficientnet_b5_rr','se_resnet101_rr', 'nasnetamobile_rr',
                'efficientnet_b6_ss','resnext_101_32_8_wsl_rr','dense_169']

their_ensemble = ['efficientnet_b0_rr','efficientnet_b1_rr','efficientnet_b2_rr','efficientnet_b3_rr',
                  'efficientnet_b4_rr','efficientnet_b5_rr','efficientnet_b0_ss','efficientnet_b1_ss',
                  'efficientnet_b2_ss','efficientnet_b3_ss','efficientnet_b4_ss','efficientnet_b5_ss',
                  'efficientnet_b6_ss', 'senet154_ss']
show_ISIC_results = False
show_ISIC_AISC_results = False
create_official_test = False
for_sjov = False

if for_sjov:

    isic_zero = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic.csv')
    isic_aisc0 = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic0.csv')
    isic_aisc1 = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic1.csv')
    isic_aisc2 = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic2.csv')
    isic_aisc3 = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic3.csv')
    isic_aisc4 = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic4.csv')
    ensemble = pd.read_csv(r'C:\Users\ptrkm\Bachelor\official_ensemble.csv')


if create_official_test:
    name = "res101_isic"
    path = r"C:\Users\ptrkm\Bachelor"
    os.chdir(path)
    pcl = pickle.load(open(r'C:\Users\ptrkm\Bachelor\2019_rr.resnet_101_rr_AISC_ISIC\2019_rr'
                           r'.resnet101_rr_AISC_ISIC_pt2\CV.pkl', "rb"))
    labels_isic = pd.read_csv(
        r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_2019_Training_GroundTruth.csv')
    image_names = os.listdir(r'C:\Users\ptrkm\Bachelor\Test Data ISIC\Cropped_p2')
    image_names = [i.removesuffix('.jpg') for i in image_names]
    official_frame = pd.DataFrame()
    official_frame['image'] = image_names

    if len(pcl['extPred'])==1:

        for j,i in enumerate(labels_isic.columns):
            if 'image' not in i:
                official_frame[i] = pcl['extPred'][0][:,j-1].tolist()
        official_frame.to_csv(os.path.join(r'C:\Users\ptrkm\Bachelor',name)+'.csv',index=False)
    else:
        our_ensemble = np.zeros(pcl['extPred'][0].shape)
        for fold in range(len(pcl['extPred'])):
            our_ensemble += pcl['extPred'][fold]/len(pcl['extPred'])
            for j, i in enumerate(labels_isic.columns):
                if 'image' not in i:
                    if 'UNK' in i:
                        official_frame[i] = np.zeros((len(official_frame),1))
                    else:
                        official_frame[i] = pcl['extPred'][fold][:,j-1].tolist()

            official_frame.to_csv(os.path.join(r'C:\Users\ptrkm\Bachelor',name+str(fold))+'.csv',index=False)
        official_ensemble = pd.DataFrame()

        for j,i in enumerate(official_frame.columns):
            if 'image' in i:
                official_ensemble[i] = official_frame[i]
            elif 'UNK' in i:
                official_ensemble[i] = np.zeros((len(official_ensemble),1))
            else:
                official_ensemble[i] = our_ensemble[:,j-1]

        official_ensemble.to_csv(os.path.join(r'C:\Users\ptrkm\Bachelor', 'official_ensemble.csv'),index=False)












if show_ISIC_results:
    os.chdir(r'C:\Users\ptrkm\Bachelor')
    pcl = pickle.load(open(r'2019_rr.resnet101_rr_AISC_gpu0_58_predn.pkl', "rb"))
    labels_isic = pd.read_csv(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_2019_Training_GroundTruth.csv')
    labels_isic.drop(['image'], axis = 1, inplace = True)
    labels_array = labels_isic.values
    weighted_accuracy = []
    accuracy = []

    for i in range(len(pcl['extPred'])):
        if i == 2:
            weighted_accuracy.append(metrics.balanced_accuracy_score(
                np.argmax(labels_array, 1), np.argmax(pcl['extPred'][i], 1)))
            accuracy.append(metrics.accuracy_score(
                np.argmax(labels_array, 1), np.argmax(pcl['extPred'][i], 1)))
            conf_mat = metrics.confusion_matrix(np.argmax(labels_array, 1),np.argmax(pcl['extPred'][i],1))
            print("Confusion matrix for evaluation on ISIC dataset trained on AISC")
            print(conf_mat)
            print("Per class weighted accuracy")
            print(conf_mat.diagonal()/np.sum(conf_mat,axis =1))

    print("weighted accuracy")
    print(weighted_accuracy)
    print("accuracy")
    print(accuracy)
    # print(np.mean(weighted_accuracy))
    # print(np.mean(accuracy))
    #
    # breakpoint()

if show_ISIC_AISC_results:
    os.chdir(r'C:\Users\ptrkm\Bachelor')

    # l_aisc = pd.read_csv(r'AISC_val_im_uids.txt',header = None)
    # breakpoint()
    pcl = pickle.load(open(r'C:\Users\ptrkm\Bachelor\2019_rr.resnet_101_rr_AISC_ISIC\2019_rr'
                           r'.resnet101_rr_AISC_ISIC_pt2\CV.pkl', "rb"))
    labels_isic_aisc = pd.read_csv(r'labels_aisc_isic.csv')
    indices = pickle.load(open(r'indices_aisc_plus_isic.pkl', "rb"))
    indices_val = indices['valIndCV']

    l_isic = [j for j,i in enumerate(labels_isic_aisc['image'][indices_val]) if 'ISIC' in i]
    l_aisc = [j for j,i in enumerate(labels_isic_aisc['image'][indices_val]) if 'ISIC' not in i]
    print((len(l_aisc),len(l_isic)))
    # breakpoint()
    # print(np.argmax(labels_isic_aisc.values[indices_val,:],axis = 1)-np.argmax(pcl['targets'][0],axis = 1))
    # breakpoint()
    labels_isic_aisc.drop(['image'], axis=1, inplace=True)
    labels_array = pcl['targets'][0]
    predictions = pcl['bestPred'][0]
    predictions_aisc = predictions[l_aisc]
    predictions_isic = predictions[l_isic]
    targets_aisc = pcl['targets'][0][l_aisc]
    targets_isic = pcl['targets'][0][l_isic]

    # print(metrics.confusion_matrix(np.argmax(pcl['targets'][0],axis = 1), np.argmax(pcl['bestPred'][0],axis=1)))

    conf_mat_isic = metrics.confusion_matrix(np.argmax(targets_isic,axis = 1), np.argmax(predictions_isic,axis=1))
    conf_mat_aisc = metrics.confusion_matrix(np.argmax(targets_aisc,axis = 1), np.argmax(predictions_aisc,axis=1))
    conf_mat_tots = metrics.confusion_matrix(np.argmax(labels_array,axis = 1), np.argmax(predictions,axis=1))

    weight_acc_isic = metrics.balanced_accuracy_score(np.argmax(targets_isic,axis = 1), np.argmax(predictions_isic,axis=1))
    weight_acc_aisc = metrics.balanced_accuracy_score(np.argmax(targets_aisc, axis=1), np.argmax(predictions_aisc, axis=1))
    weight_acc_tots = metrics.balanced_accuracy_score(np.argmax(labels_array, axis=1), np.argmax(predictions, axis=1))

    acc_isic = metrics.accuracy_score(np.argmax(targets_isic,axis = 1), np.argmax(predictions_isic,axis=1))
    acc_aisc = metrics.accuracy_score(np.argmax(targets_aisc, axis=1), np.argmax(predictions_aisc, axis=1))
    acc_tots = metrics.accuracy_score(np.argmax(labels_array, axis=1), np.argmax(predictions, axis=1))

    pr_class_isic = conf_mat_isic.diagonal()/np.sum(conf_mat_isic, axis= 1)
    pr_class_aisc = conf_mat_aisc.diagonal() /np.sum(conf_mat_aisc, axis=1)
    pr_class_tots = conf_mat_tots.diagonal()/np.sum(conf_mat_tots,axis = 1)


    print("Results on the combined ISIC AISC dataset")
    print("Confusion matrix on evaluation on ISIC:")
    print(conf_mat_isic)
    print("Per class weighted accuracy on ISIC")
    print(pr_class_isic)
    print("Weighted accuracy on isic:")
    print(weight_acc_isic)
    print("#####################################################")
    print("Confusion matrix on evaluation on AISC:")
    print(conf_mat_aisc)
    print("Per class weighted accuracy on aisc")
    print(pr_class_aisc)
    print("Weighted accuracy on aisc:")
    print(weight_acc_aisc)
    print("#####################################################")

    print("Confusion matrix on evaluation on combined:")
    print(conf_mat_tots)
    print("Per class weighted accuracy on combined")
    print(pr_class_tots)
    print("Weighted accuracy on combined:")
    print(weight_acc_tots)



    breakpoint()







if show_AISC_results:
    results_pcl = {}
    labels_frame = pd.read_csv(r'C:\Users\ptrkm\Bachelor\labels.csv')

    results_pcl['ground_truth'] = labels_frame
    images = labels_frame['image']
    labels_array = labels_frame.drop(['image'], axis = 1).values
    labels_array = labels_array[:,:-1]
    our_accuracies = []
    our_weighted_accuracies = []

    their_accuracies = []
    their_weighted_accuracies = []
    if average_voting:
        their_ensemble_predictions = np.zeros(labels_array.shape)
        our_ensemble_predictions = np.zeros(labels_array.shape)
    else:
        their_ensemble_predictions = np.zeros((labels_array.shape[0]-1, len(their_ensemble)))
        our_ensemble_predictions = np.zeros((labels_array.shape[0]-1, len(our_ensemble)))
        index_our = 0
        index_their = 0
    for file in os.listdir(path):

        if "pkl" in file:
            for i in our_ensemble:
                if i in file:
                    pcl = pickle.load(open(os.path.join(path,file), "rb"))
                    predictions = pcl['extPred'][0]
                    if predictions.shape[1] == 9:
                        predictions = predictions[:,:-1]

                    our_weighted_accuracies.append((i,metrics.balanced_accuracy_score(
                        np.argmax(labels_array, 1), np.argmax(predictions,1))))
                    our_accuracies.append((i,metrics.accuracy_score(
                        np.argmax(labels_array, 1), np.argmax(predictions,1))))
                    if average_voting:
                        our_ensemble_predictions+= predictions*1/len(our_ensemble)
                    # insert case for majority voting
                    else:
                        our_ensemble_predictions[:,index_our] = (np.argmax(predictions,1)).astype(int)
                        index_our+= 1

            for i in their_ensemble:
                if i in file:
                    pcl = pickle.load(open(os.path.join(path,file), "rb"))
                    predictions = pcl['extPred'][0]
                    if predictions.shape[1] == 9:
                        predictions = predictions[:,:-1]
                    their_weighted_accuracies.append((i,metrics.balanced_accuracy_score(
                        np.argmax(labels_array, 1), np.argmax(predictions,1))))
                    their_accuracies.append((i,metrics.accuracy_score(
                        np.argmax(labels_array, 1), np.argmax(predictions,1))))
                    if average_voting:
                        their_ensemble_predictions+= predictions*1/len(our_ensemble)
                    else:
                        their_ensemble_predictions[:,index_their] = (np.argmax(predictions,1)).astype(int)
                        index_their+=1
    if not average_voting:

        our_majority = [np.bincount(our_ensemble_predictions[i,:].astype(int)).argmax() for i in range(labels_array.shape[0])]
        their_majority = [np.bincount(their_ensemble_predictions[i,:].astype(int)).argmax() for i in range(labels_array.shape[0])]
        our_ensemble_predictions = our_majority
        their_ensemble_predictions = their_majority

        our_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array,1), our_ensemble_predictions)))

        our_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array,1), our_ensemble_predictions)))

        their_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array,1), their_ensemble_predictions)))

        their_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array,1),their_ensemble_predictions)))

        our_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array,1),our_ensemble_predictions)
        their_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array, 1), their_ensemble_predictions)
    else:
        our_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))))

        our_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))))

        their_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array, 1), np.argmax(their_ensemble_predictions, 1))))

        their_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array, 1), np.argmax(their_ensemble_predictions, 1))))

        our_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))
        their_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array, 1),
                                                       np.argmax(their_ensemble_predictions, 1))
    nuestro_ensemble = pd.DataFrame()
    sus_ensemble = pd.DataFrame()

    for idx, columns in enumerate(labels_frame.columns):
        if columns == 'image':
            nuestro_ensemble[columns] = images
            sus_ensemble[columns] = images
        elif columns!='UNK':
            nuestro_ensemble[columns] = our_ensemble_predictions[:,idx-1]
            sus_ensemble[columns] = their_ensemble_predictions[:,idx-1]

    results_pcl['our_ensemble_predictions'] = nuestro_ensemble
    results_pcl['their_ensemble_predictions'] = sus_ensemble

    with open(r'C:\Users\ptrkm\Bachelor\ensemble_results.pkl','wb') as handle:
        pickle.dump(results_pcl,handle, protocol=pickle.HIGHEST_PROTOCOL)
    breakpoint()
    our_weighted_accuracies = dict(our_weighted_accuracies)
    our_accuracies = dict(our_accuracies)
    their_weighted_accuracies = dict(their_weighted_accuracies)
    their_accuracies = dict(their_accuracies)
    print("RESULTS ON AISC DATASET")
    print("our ensemble config confusion matrix")
    print(our_confusion_mat)
    print('##########################################')
    print("Their ensemble config confusion matrix")
    print(our_confusion_mat)
    print('##########################################')
    print("Our ensemble weighted accuracy: " + str(our_weighted_accuracies['ensemble']), end = "", flush=True)
    print("    Their ensemble weighted accuracy: " + str(their_weighted_accuracies['ensemble']))
    print("Our ensemble accuracy: " + str(our_accuracies['ensemble']),end = "", flush=True)
    print("    Their ensemble accuracy: " + str(their_accuracies['ensemble']))




if show_training_results:
    weighted_accuracy = []
    sensitivity = []
    auc = []
    networks = []
    path_rr = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Ny mappe'
    path_ss = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning'
    # T = io.loadmat(path_ss + '\\progression_valInd.mat')
    # weighted_accuracy.append(metrics.balanced_accuracy_score(np.argmax(pcl['targets'][0],1),np.argmax(pcl['bestPred'][0],1)))
    idx = 0

    our_accuracies = []
    our_weighted_accuracies = []

    their_accuracies = []
    their_weighted_accuracies = []

    for i in os.listdir(path_ss):
        if '2019' in i and 'ss' in i:
            for ens_our in our_ensemble:
                if ens_our in i:
                    next_path = os.path.join(path_ss,i)
                    if "CV.pkl" in os.listdir(next_path):
                        os.chdir(next_path)
                        pcl = pickle.load(open(r'CV.pkl', "rb"))
                        predictions = pcl['bestPred'][0]
                        labels_array = pcl['targets'][0]
                        if idx == 0:
                            our_ensemble_predictions = np.zeros(labels_array.shape)
                            their_ensemble_predictions = np.zeros(labels.array.shape)

                        our_weighted_accuracies.append((i,metrics.balanced_accuracy_score(
                            np.argmax(labels_array, 1), np.argmax(predictions,1))))
                        our_accuracies.append((i,metrics.accuracy_score(
                            np.argmax(labels_array, 1), np.argmax(predictions,1))))

                        if average_voting:
                            our_ensemble_predictions += predictions * 1 / len(our_ensemble)

            for ens_their in their_ensemble:
                if ens_their in i:
                    next_path = os.path.join(path_ss, i)
                    if "CV.pkl" in os.listdir(next_path):
                        os.chdir(next_path)
                        pcl = pickle.load(open(r'CV.pkl', "rb"))
                        predictions = pcl['bestPred'][0]
                        labels_array = pcl['targets'][0]
                        if idx == 0:
                            their_ensemble_predictions = np.zeros(labels_array.shape)
                            our_ensemble_predictions = np.zeros(labels_array.shape)
                            idx+=1
                        our_weighted_accuracies.append((i, metrics.balanced_accuracy_score(
                            np.argmax(labels_array, 1), np.argmax(predictions, 1))))
                        our_accuracies.append((i, metrics.accuracy_score(
                            np.argmax(labels_array, 1), np.argmax(predictions, 1))))

                        if average_voting:
                            their_ensemble_predictions += predictions * 1 / len(our_ensemble)
    for i in os.listdir(path_rr):
        if '2019' in i and 'rr' in i:
            for ens_our in our_ensemble:
                if ens_our in i:
                    next_path = os.path.join(path_rr,i)
                    if "CV.pkl" in os.listdir(next_path):
                        os.chdir(next_path)
                        pcl = pickle.load(open(r'CV.pkl', "rb"))
                        predictions = pcl['bestPred'][0]
                        labels_array = pcl['targets'][0]
                        if labels_array.shape[0] < 10000:
                            our_weighted_accuracies.append((i,metrics.balanced_accuracy_score(
                                np.argmax(labels_array, 1), np.argmax(predictions,1))))
                            our_accuracies.append((i,metrics.accuracy_score(
                                np.argmax(labels_array, 1), np.argmax(predictions,1))))

                            if average_voting:
                                our_ensemble_predictions += predictions * 1 / len(our_ensemble)

            for ens_their in their_ensemble:
                if ens_their in i:

                    next_path = os.path.join(path_rr, i)
                    if "CV.pkl" in os.listdir(next_path):
                        os.chdir(next_path)
                        pcl = pickle.load(open(r'CV.pkl', "rb"))
                        predictions = pcl['bestPred'][0]
                        labels_array = pcl['targets'][0]
                        if labels_array.shape[0] < 10000:
                            our_weighted_accuracies.append((i, metrics.balanced_accuracy_score(
                                np.argmax(labels_array, 1), np.argmax(predictions, 1))))
                            our_accuracies.append((i, metrics.accuracy_score(
                                np.argmax(labels_array, 1), np.argmax(predictions, 1))))

                            if average_voting:
                                their_ensemble_predictions += predictions * 1 / len(our_ensemble)
    if average_voting:
        our_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))))


        our_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))))

        their_weighted_accuracies.append(('ensemble', metrics.balanced_accuracy_score(
            np.argmax(labels_array, 1), np.argmax(their_ensemble_predictions, 1))))

        their_accuracies.append(('ensemble', metrics.accuracy_score(
            np.argmax(labels_array, 1), np.argmax(their_ensemble_predictions, 1))))

        our_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array, 1), np.argmax(our_ensemble_predictions, 1))
        their_confusion_mat = metrics.confusion_matrix(np.argmax(labels_array, 1),
                                                       np.argmax(their_ensemble_predictions, 1))

    our_weighted_accuracies = dict(our_weighted_accuracies)
    our_accuracies = dict(our_accuracies)
    their_weighted_accuracies = dict(their_weighted_accuracies)
    their_accuracies = dict(their_accuracies)
    print("RESULTS FROM VALIDATION DURING TRAINING ON ISIC")
    print("Our confusion matrix")
    print(our_confusion_mat)
    print("Their ensemble confusion matrix")
    print(their_confusion_mat)
    print('##########################################')
    print('##########################################')

    print("Our ensemble config. weighted accuracy: " + str(our_weighted_accuracies['ensemble']), end = "", flush=True)
    print("    Their ensemble config. weighted accuracy: " + str(their_weighted_accuracies['ensemble']))
    print("Our ensemble config. accuracy: " + str(our_accuracies['ensemble']), end = "", flush = True)
    print("    Their ensemble config. accuracy: " + str(their_accuracies['ensemble']))











    # collected['networks'] = networks
    # # collected['weighted_accuracy'] = weighted_accuracy
    # collected['auc'] = auc
    # collected['sensitivity'] = sensitivity
    # os.chdir(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\intermediate plots')
    #
    #
    #
    # import matplotlib.backends.backend_pdf
    # pdf = matplotlib.backends.backend_pdf.PdfPages("Results1.pdf")
    # for j, i in enumerate(collected.columns):
    #     if 'network' not in i:
    #         figure = plt.figure(figsize=(12,9))
    #         plt.bar(collected['networks'],collected[i])
    #         plt.xlabel('networks')
    #         plt.xticks(rotation = 90)
    #         plt.ylabel(i)
    #         plt.title('best ' + i + ' for same sized cropping')
    #         pdf.savefig(figure)
    # pdf.close()
    #
    #
    # collected.to_csv(r'classification.csv')
    # breakpoint()
    #





