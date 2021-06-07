import os
import pandas as pd

# path = r"\scratch\s184400\test_a\\"
directory_path = r"C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\WBInput"
# save_path = r"\scratch\s184400\directory_ordering.csv"
save_path = directory_path + "\directory_ordering.csv"
os.chdir(directory_path)
image_names = os.listdir()
pd_frame = pd.DataFrame()

pd_frame['images'] = image_names

pd_frame.to_csv(save_path)
