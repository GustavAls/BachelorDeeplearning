
import os
import shutil
import pickle

os.chdir(r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\\')
directories = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Ny mappe\\'
target_folder = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\CV_results\\'

folders = os.listdir(os.getcwd())

for i in folders:
    if "2019" in i and ".pkl" not in i:
        new_file_name = i.removeprefix(directories)
        new_file_name = new_file_name.removeprefix("2019.test_")

        if "CV.pkl" in os.listdir(i):
            original = i+"\\CV.pkl"
            target_file = target_folder + new_file_name + ".pkl"
            shutil.copyfile(original,target_file)



