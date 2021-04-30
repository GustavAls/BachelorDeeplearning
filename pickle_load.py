import pickle
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Bruger\Desktop')
pcl = pickle.load(open(r'C:\Users\Bruger\Desktop\indices_isic2019_one_cv.pkl', "rb"))
# pcl2 = pickle.load(open(r'C:\Users\Bruger\Desktop\indices_isic2019.pkl', 'rb'))

# isic_train = pcl2['trainIndCV'][0]
# isic_val = pcl2['valIndCV'][0]


train_ind = pcl['trainIndCV']
val_ind = pcl['valIndCV']

# print(len(np.intersect1d(isic_train, train_ind)))
# print(len(np.intersect1d(isic_val, val_ind)))
# print(len(val_ind))

        
breakpoint()

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
