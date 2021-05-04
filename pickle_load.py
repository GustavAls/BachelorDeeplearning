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

pcl = pickle.load(open(r'2019_rr.test_res101_rr_bestgpu1_130_predn.pkl', "rb"))
labels = pd.read_csv(r'labels.csv')
predictions = []
labels_new = np.zeros(pcl['extPred'][0].shape)
labels_list = list(labels['image'])
labels_frame = labels.drop(['image'],axis=1).values
pcl['all_images'][0] = pd.unique(pcl['all_images'][0])
preds_new = []
labels_new = []
for idx,ims in enumerate(pcl['all_images'][0]):
    if '.jpeg' not in ims:
        image = ims.removesuffix('.jpg')
        image = image.removeprefix('/home/s184400/isic2019/AISC_images/official/')
        labels_new.append(labels_frame[labels_list.index(image)])
        preds_new.append(pcl['extPred'][0][idx])

    # else:
    #     image = ims.removesuffix('.jpeg')
    #     image = image.removeprefix('/home/s184400/isic2019/AISC_images/official/')
    #     labels_new[idx] = labels_frame[labels_list.index(image)]

print(metric.accuracy_score(np.argmax(np.array(labels_new),axis=1),np.argmax(np.array(preds_new),axis=1)))
print(metric.balanced_accuracy_score(np.argmax(np.array(labels_new),axis=1),np.argmax(np.array(preds_new),axis=1)))

breakpoint()

print(metric.accuracy_score(np.argmax(labels_new,axis=1),np.argmax(pcl['extPred'][0],axis=1)))
print(metric.balanced_accuracy_score(np.argmax(labels_new,axis=1),np.argmax(pcl['extPred'][0],axis=1)))


breakpoint()

training_inds = pcl['trainIndCV'].tolist()
val_inds = pcl['valIndCV'].tolist()
training_inds = []
for i in range(len(labels)):
    if i not in val_inds:
        training_inds.append(i)
training_inds = np.array(training_inds)
val_inds = np.array(val_inds)

pcl['trainIndCV'] = training_inds
pcl['valIndCV'] = val_inds

with open('indices_aisc_plus_isic.pkl','wb') as handle:
    pickle.dump(pcl, handle, protocol=pickle.HIGHEST_PROTOCOL)
breakpoint()



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
