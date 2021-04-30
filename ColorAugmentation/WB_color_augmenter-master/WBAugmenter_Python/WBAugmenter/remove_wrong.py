


import os
import sys

input_path = sys.argv[1]

files_in_path = os.listdir(input_path)

for files in files_in_path:
    if '.jpg' in files:
        os.remove(os.path.join(input_path,files))