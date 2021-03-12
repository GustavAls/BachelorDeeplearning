
import ctypes
import platform
import os
import shutil



path_one = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\Data ISIC\ISIC_test_cropped\\'
path_two = r'C:\Users\ptrkm\OneDrive\Dokumenter\TestFolder\return\\'
path_three = r'D:\Data\\'
list_one = os.listdir(path_one)
list_two = os.listdir(path_two)
index = 0
# for i, j in enumerate(list_one):
#
#     if j == "ISIC_0033057.jpg":
#         index = i
breakpoint()
for i, j in enumerate(list_one):

        try:
            original = path_two + j
            target = path_three + j
            shutil.copyfile(original,target)
        except:
            print("USB is full, we reached " + j)
            print("File is number" + str(i))
            break





