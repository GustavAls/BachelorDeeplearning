import numpy as np
import cv2
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

image_path = r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\WBAugmented'
os.chdir(image_path)
image_files = os.listdir(image_path)
list_org_index = [i for i,j in enumerate(image_files) if 'original' in j]
for i, org_idx in enumerate(list_org_index):
    first_column_index = 6 * i
    if i == 0:
        image_files[i], image_files[org_idx] = image_files[org_idx], image_files[i]
    else:
        image_files[first_column_index], image_files[org_idx] = image_files[org_idx], image_files[first_column_index]
images = []

for file in image_files:
    images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))

image_height, image_width, _ = images[0].shape

height = int(image_height / 5)
width = int(image_width / 5 )
idx = 0
canvas = np.zeros((height * 4, width * 6, 3))

for i in range(4):
    for j in range(6):
        print("IDX: " + str(idx))
        resized_image = cv2.resize(images[idx], (width, height), interpolation=cv2.INTER_AREA)
        canvas[i * height:(i+1) * height, j * width:(j+1) * width,:] = np.array(resized_image).astype('uint8')
        idx += 1


plt.figure()
plt.imshow(canvas.astype('uint8'))
# Turn off tick labels
plt.axis('off')
plt.show()
plt.savefig(r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\ISBI2016_ISIC_Part2B_Training_Data\\AugmentedImages.png', format='png', pad_inches=0)


