

import pandas as pd
import numpy as np
import os
import pickle
import sklearn.metrics as metrics

best_res = pd.read_csv(r'C:\Users\ptrkm\Bachelor\res101_isic0.csv')
bad_res = pickle.load(open(r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_ISIC_AISC_on_ISIC2019\2019_mixed.res101_rr_lastgpu6_100_predn.pkl','rb'))
image_list = bad_res['all_images'][0]
image_list = [i.removeprefix('/home/s184400/isic2019/ISIC_test_cropped/') for i in image_list]
image_list = [i.removesuffix('.jpg') for i in image_list]

preds = bad_res['extPred'][0]
results_list = []
breakpoint()
for image in best_res['image']:
    index = image_list.index(image)
    results_list.append(preds[index])
results_list = np.asarray(results_list)

print(metrics.confusion_matrix(np.argmax(best_res.drop(['image'],axis=1).values,axis=1),np.argmax(results_list,axis=1)))


