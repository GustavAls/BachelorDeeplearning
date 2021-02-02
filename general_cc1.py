import numpy as np
import argparse
import cv2


def max_rgb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)
    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


# construct the argument parse and parse the arguments

image = cv2.imread(r'C:\Users\ptrkm\Downloads\Billedanalyse\Billedanalyse\Exercises\DTUSign1.jpg', 1)
filtered = max_rgb_filter(image)

cv2.imshow('image1',filtered)
cv2.waitKey(0)
