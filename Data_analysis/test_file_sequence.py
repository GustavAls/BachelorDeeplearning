

import os
import pickle
import sys
system_arguments = sys.argv
os.chdir(r'C:\Users\ptrkm\Bachelor')
pcl = pickle.load(open())


pcl = {}

pcl['file_sequence'] = os.listdir(system_arguments[1])

with open(os.path.join(system_arguments[2],'file_sequence.pkl'),'wb') as handle:
    pickle.dump(pcl, handle, protocol=pickle.HIGHEST_PROTOCOL)





