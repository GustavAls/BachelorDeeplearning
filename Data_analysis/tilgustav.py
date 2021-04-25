
import pickle
import pandas as pd
import os
import numpy as np


os.chdir(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC')
path = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning'

pcl = pickle.load(open(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\indices_isic2019.pkl', "rb"))

frame = pd.read_csv(r'ISIC_2019_Training_GroundTruth.csv')

for i in frame.columns:
    print(np.sum(frame[i][pcl['trainIndCV'][0]]))


breakpoint()


