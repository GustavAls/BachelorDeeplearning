import pickle
import pandas as pd

p = pickle.load(open(r'C:\Users\Bruger\Desktop\CV.pkl', "rb"))
panda_p = pd.read_pickle(r'C:\Users\Bruger\Desktop\CV.pkl')
f = open("dict.txt","w")
f.write( str(panda_p) )
f.close
