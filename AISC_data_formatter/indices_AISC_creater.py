
import sklearn.model_selection as model_selection
import numpy as np
import pandas as pd
import os
import pickle
os.chdir(r'C:\Users\ptrkm\Bachelor')
df = pd.read_csv(r'labels.csv')
df['UNK'] = [0.0 for i in range(len(df))]
#
# df.drop(['Unnamed: 0'],axis = 1, inplace=True)
# df.to_csv(r'labels.csv',index=False)

os.chdir(r'C:\Users\ptrkm\OneDrive\Skrivebord\isic2019\labels\official')
df_isic = pd.read_csv(r'labels.csv')

os.chdir(r'C:\Users\ptrkm\Bachelor')
cross = model_selection.KFold(n_splits=5, random_state=7, shuffle= True)
test_inds= []
train_inds = []
write_AISC = False
write_AISC_plus_isic = True

if write_AISC:
    for idx, i in enumerate(df.columns):
        if "image" not in i:
            inds = np.where(df[i] == 1)[0]
            for j, (train, test) in enumerate(cross.split(inds)):
                if idx == 1:
                    train_inds.append(inds[train])
                    test_inds.append(inds[test])
                else:
                    train_inds[j] = np.append(train_inds[j], inds[train])
                    test_inds[j] = np.append(test_inds[j], inds[test])

    pcl = {'trainIndCV': train_inds,'valIndCV':test_inds}

    for i in range(5):
        print(np.intersect1d(pcl['trainIndCV'][i],pcl['valIndCV'][i]))

    # with open(r'indices_AISC.pkl','wb') as handle:
    #     pickle.dump(pcl,handle, protocol=pickle.HIGHEST_PROTOCOL)


df_is_aisc = pd.concat([df,df_isic],axis = 0,ignore_index=True)

from sklearn.utils import shuffle
df_is_aisc = shuffle(df_is_aisc)
df_is_aisc.reset_index(inplace = True, drop = True)
test_inds= []
train_inds = []
if write_AISC_plus_isic:
    for idx, i in enumerate(df_is_aisc.columns):
        if "image" not in i and "UNK" not in i:
            inds = np.where(df_is_aisc[i] == 1)[0]
            for j, (train, test) in enumerate(cross.split(inds)):
                if idx == 1:
                    train_inds.append(inds[train])
                    test_inds.append(inds[test])
                else:
                    train_inds[j] = np.append(train_inds[j], inds[train])
                    test_inds[j] = np.append(test_inds[j], inds[test])

    pcl = {'trainIndCV': train_inds,'valIndCV':test_inds}

    for i in range(5):
        print(np.intersect1d(pcl['trainIndCV'][i],pcl['valIndCV'][i]))

    with open(r'indices_aisc_plus_isic.pkl','wb') as handle:
        pickle.dump(pcl,handle, protocol=pickle.HIGHEST_PROTOCOL)
    df_is_aisc.to_csv(r'labels_aisc_isic.csv',index=False)

breakpoint()
