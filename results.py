import numpy as np
from scipy import io
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import random
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score, balanced_accuracy_score
os.chdir(r'C:\Users\Bruger\Desktop')
wb_aisc = True
aisc = False
plot_roc = True
write_to_pkl = False

if wb_aisc:
    pcl = pickle.load(open(r'C:\Users\Bruger\Desktop\2019_rr.resnet101_rr_wb_bestgpu4_30_predn.pkl', "rb"))
    labels = pd.read_csv(r'labels.csv')
    diagnosis = labels.columns
    diagnosis = np.asarray(diagnosis[1:9])
    labels = labels.drop(columns=['image']).to_numpy()
    predictions = pcl['extPred'][0]
    averaged_predictions = []
    temp_prediction = []
    for idx, prediction in enumerate(predictions):
        temp_prediction.append(prediction)
        if (idx + 1) % 6 == 0:
            temp_prediction = np.mean(temp_prediction, axis=0)
            averaged_predictions.append(temp_prediction)
            temp_prediction = []
elif aisc:
    pcl = pickle.load(open(r'C:\Users\Bruger\Desktop\2019_rr.resnet101_rr_AISC_gpu0_58_predn.pkl', "rb"))
    labels = pd.read_csv(r'ISIC_2019_Training_GroundTruth.csv')
    diagnosis = labels.columns
    diagnosis = np.asarray(diagnosis[1:9])
    labels = labels.drop(columns=['image']).to_numpy()
    predictions = pcl['extPred'][0]



if wb_aisc:
    predictions = np.asarray(averaged_predictions)

y_pred_pre_average = np.argmax(averaged_predictions)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(labels, axis=1)

weighted_accuracy = balanced_accuracy_score(y_true, y_pred)
confusion_matrix = confusion_matrix(y_true, y_pred)
weighted_accuracies = confusion_matrix.diagonal() / np.sum(confusion_matrix, axis=1)
accuracy = accuracy_score(y_true, y_pred)
fpr = {}
tpr = {}
roc_auc = np.zeros([8])
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

auc = np.mean(roc_auc)
std_auc = np.std(roc_auc)

new_tpr = []
new_fpr = []

lengths = []

for i in range(len(fpr)):
    lengths.append(len(fpr[i]))
length_smallest_class = np.min(lengths)

np.random.seed(1)
for i in range(len(fpr)):
    new_tpr.append(sorted(sample(list(tpr[i]), length_smallest_class)))
    new_fpr.append(sorted(sample(list(fpr[i]), length_smallest_class)))

new_tpr = np.asarray(new_tpr)
new_fpr = np.asarray(new_fpr)

mean_tpr = np.mean(new_tpr, axis=0)
mean_fpr = np.mean(new_fpr, axis=0)

if wb_aisc:
    print("Result for applying WB to ISIC")
elif aisc:
    print("Result for evaluating an AISC trained ResNet101 on ISIC:")
print("Pr class weighted accuracy: " + str(weighted_accuracies))
print("Mean Weighted accuracy " + str(weighted_accuracy))
print("Accuracy: " + str(accuracy))
print("Pr class AUC: " + str(roc_auc))
print("Mean AUC: " + str(auc))
print("Confusion matrix:")
print(confusion_matrix)
if plot_roc:
    plt.figure()
    linewidth = 1.2
    sns.set_style("whitegrid")
    sns.set_theme()
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], '-', linewidth=linewidth, label=diagnosis[i], alpha=0.6)
    plt.plot(mean_fpr, mean_tpr, color='blue',
        label=r"Mean ROC" "\n" "(AUC = %0.2f $\pm$ %0.2f)" % (auc, std_auc),
        lw=2, alpha=.95)
    plt.plot([0, 1], [0, 1], color='red', lw=linewidth, linestyle='--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=9)
    plt.show()


if write_to_pkl:
    if wb_aisc:
        pcl_new = {'extPred':predictions}

        with open('wb_aisc_pred.pkl','wb') as handle:
            pickle.dump(pcl_new,handle, protocol=pickle.HIGHEST_PROTOCOL)


breakpoint()