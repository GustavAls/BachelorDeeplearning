import pickle
import pandas as pd
import numpy as np
import os
import sklearn.metrics as metric
os.chdir(r'C:\Users\ptrkm\Bachelor')

# import json
#
# with open('images.json') as json_file:
#     data = json.load(json_file)
#
# breakpoint()

#

pcl = pickle.load(open(r'indices_aisc_plus_isic.pkl', "rb"))
labels = pd.read_csv(r'labels_aisc_isic.csv')

aisc_training = []
aisc_test =[]

for idx, label in enumerate(labels['image']):
    if 'ISIC' not in label and idx in pcl['trainIndCV']:
        aisc_training.append(label)
    elif 'ISIC' not in label and idx in pcl['valIndCV']:
        aisc_test.append(label)
training_ims = pd.DataFrame()
validation_ims = pd.DataFrame()
training_ims['images'] = aisc_training
validation_ims['images'] = aisc_test
training_ims.to_csv('images_for_train.txt',header = False,index = False)
validation_ims.to_csv('images_for_val.txt',header = False,index = False)


breakpoint()

labels_frame = pd.read_csv(r'C:\Users\ptrkm\Bachelor\labels.csv')






labels_frame = labels_frame.drop(['image'],axis = 1)
label_array = labels_frame.values
weighted_accuracy = (metric.accuracy_score(np.argmax(label_array,1),np.argmax(pcl['extPred'][0],1)))
print(weighted_accuracy)
weighted_accuracy = metric.balanced_accuracy_score(np.argmax(label_array,1),np.argmax(pcl['extPred'][0],1))
print(weighted_accuracy)
confusion_matrix = (metric.confusion_matrix(np.argmax(label_array,1),np.argmax(pcl['extPred'][0],1)))
print(confusion_matrix)

# indices = []
# for i in range(label_array.shape[0]):
#     if label_array[i,-1] == 1:
#         indices.append(i)
# for_fun_label = np.delete(label_array, indices, axis = 0)
# for_fun_label = for_fun_label[:,:7]
# for_fun_pred = pcl['extPred'][0][:,:7]
# for_fun_pred = np.delete(for_fun_pred,indices,axis= 0)
# weighted_accuracy = metric.balanced_accuracy_score(np.argmax(for_fun_label,1),np.argmax(for_fun_pred,1))
# print(weighted_accuracy)
breakpoint()


#
# print(pcl.keys())



breakpoint()
#
#
# train_ind = pcl['trainIndCV'].tolist()
# val_ind = pcl['valIndCV'].tolist()
#
# for i in val_ind:
#     if i not in train_ind:
#         train_ind.append(i)
#     else:
#         print(i + 'was already here')
#
# pcl_new = {'trainIndCV':np.asarray(train_ind),'valIndCV':np.asarray(train_ind)}
#
# with open('new_pickle.pkl','wb') as handle:
#     pickle.dump(pcl_new,handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('new_pickle.pkl','rb') as handle:
#     test = pickle.load(handle)


breakpoint()
