

import numpy as np
import pandas as pd
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import psutil
import matplotlib.pyplot as plt
import os
import pickle
from sklearn import metrics
from scipy import io

path = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\\'
collected = pd.DataFrame()

weighted_accuracy = []
sensitivity = []
auc = []
networks = []

for i in os.listdir(path):
    if '2019' in i and 'ss' in i:
        path_next = path + i
        if 'CVSet0' in os.listdir(path_next):

            path_next = path_next+'\\CVSet0'
            if 'progression_valInd.mat' in os.listdir(path_next):
                T = io.loadmat(path_next +'\\progression_valInd.mat')
                # weighted_accuracy.append(metrics.balanced_accuracy_score(np.argmax(pcl['targets'][0],1),np.argmax(pcl['bestPred'][0],1)))

                sensitivity.append(np.max(np.mean(T['sens'], axis = 1)))
                auc.append(np.max(np.mean(T['auc'], axis = 1)))
                network = i.replace('2019.test_','')
                network = network.replace('_ss','')
                networks.append(network)

collected['networks'] = networks
# collected['weighted_accuracy'] = weighted_accuracy
collected['auc'] = auc
collected['sensitivity'] = sensitivity
os.chdir(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\intermediate plots')



import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("Results1.pdf")
for j, i in enumerate(collected.columns):
    if 'network' not in i:
        figure = plt.figure(figsize=(12,9))
        plt.bar(collected['networks'],collected[i])
        plt.xlabel('networks')
        plt.xticks(rotation = 90)
        plt.ylabel(i)
        plt.title('best ' + i + ' for same sized cropping')
        pdf.savefig(figure)
pdf.close()


collected.to_csv(r'classification.csv')
breakpoint()






print(os.listdir(path))