import pickle
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\ptrkm\OneDrive\Skrivebord')
pcl = pickle.load(open(r'C:\Users\ptrkm\OneDrive\Skrivebord\indices_isic2019.pkl', "rb"))


train_ind = pcl['trainIndCV'].tolist()
val_ind = pcl['valIndCV'].tolist()

for i in val_ind:
    if i not in train_ind:
        train_ind.append(i)
    else:
        print(i + 'was already here')

pcl_new = {'trainIndCV':np.asarray(train_ind),'valIndCV':np.asarray(train_ind)}

with open('new_pickle.pkl','wb') as handle:
    pickle.dump(pcl_new,handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('new_pickle.pkl','rb') as handle:
    test = pickle.load(handle)


breakpoint()
