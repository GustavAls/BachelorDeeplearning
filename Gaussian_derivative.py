import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def fill_border(image, border_width):
    dimension = 1
    if len(image.shape) == 2:
        y_height, x_height = image.shape
        out_image = np.zeros((y_height + border_width * 2, x_height + border_width * 2))

    else:
        y_height, x_height, dimension = image.shape
        out_image = np.zeros((y_height + border_width * 2, x_height + border_width * 2, dimension))

    border_mat = np.ones((border_width,border_width))
    if dimension == 1:
        print('hej')
        out_image[:border_width, :border_width] = border_mat * image[0, 0]
        out_image[border_width + y_height:2 * border_width + y_height, :border_width] = border_mat * image[y_height - 1, 0]
        out_image[:border_width, border_width + x_height:2 * border_width + x_height] = border_mat * image[0, x_height - 1]
        out_image[border_width + y_height:2 * border_width + y_height, border_width + x_height:2 * border_width + x_height] = border_mat * image[y_height - 1, x_height - 1]
        # Setting the inner values equal to original image
        out_image[border_width:border_width + y_height, border_width:border_width + x_height] = image[:, :]
        # Copying and extending the values of the outer rows and columns of the original image
        print('hej')
        out_image[:border_width, border_width:border_width + x_height] = np.tile(image[0, :], (border_width, 1))
        out_image[border_width + y_height:2 * border_width + y_height, border_width:border_width + x_height] = np.tile(image[y_height - 1, :], (border_width, 1))
        out_image[border_width:border_width + y_height, :border_width] = np.transpose(np.tile(image[:, 0], (border_width, 1)))
        out_image[border_width:border_width + y_height, border_width + x_height:2 * border_width + x_height] = np.transpose(np.tile(image[:, x_height - 1], (border_width, 1)))
        print('hej')
    else:

        for i in range(dimension):
            # Setting entire corners equal to corner values in image
            out_image[:border_width,:border_width,i]=border_mat*image[0,0,i]
            out_image[border_width+y_height:2*border_width+y_height,:border_width, i]=border_mat*image[y_height-1,0,i]
            out_image[:border_width,border_width+x_height:2*border_width+x_height,i] =border_mat*image[0,x_height-1,i]
            out_image[border_width+y_height:2*border_width+y_height,border_width+x_height:2*border_width+x_height,i]=border_mat*image[y_height-1,x_height-1,i]
            # Setting the inner values equal to original image
            out_image[border_width:border_width+y_height,border_width:border_width+x_height,i]=image[:,:,i]
            # Copying and extending the values of the outer rows and columns of the original image
            out_image[:border_width,border_width:border_width+x_height,i]= np.tile(image[0,:,i],(border_width,1))
            out_image[border_width+y_height:2*border_width+y_height,border_width:border_width+x_height,i] = np.tile(image[y_height-1,:,i],(border_width,1))
            out_image[border_width:border_width+y_height,:border_width,i]=np.transpose(np.tile(image[:,0,i],(border_width,1)))
            out_image[border_width:border_width+y_height,border_width+x_height:2*border_width+x_height,i]=np.transpose(np.tile(image[:,x_height-1,i],(border_width,1)))
    print('hej')
    return out_image


""" Test of function on normal distributed data
test_im2 = np.random.normal(100,7,(50,50))
plt.figure()
plt.imshow(test_im2)
fill_border_test = fill_border(test_im2,3)
plt.figure()
plt.imshow(fill_border_test)
plt.show()
"""
"""
def fill_border(image, border_width):
    dimension = 1
    if len(image.shape)== 2:
        y_height, x_height = image.shape
        out_image = np.zeros((y_height + border_width * 2, x_height + border_width * 2, dimension))
        y_height -= 1
        x_height -= 1

    else:
        y_height, x_height, dimension = image.shape
        out_image = np.zeros((y_height + border_width * 2, x_height + border_width * 2, dimension))
        y_height -= 1
        x_height -= 1


    #border_width -= 1
    border_mat = np.ones((border_width,border_width))

    for i in range(dimension):
        # Setting entire corners equal to corner values in image
        out_image[:border_width,:border_width,i]=border_mat*image[0,0,i]
        out_image[border_width+y_height+1:2*border_width+y_height+1,:border_width, i]=border_mat*image[y_height,0,i]
        out_image[:border_width,border_width+x_height+1:2*border_width+x_height+1,i] =border_mat*image[0,x_height,i]
        out_image[border_width+y_height+1:2*border_width+y_height+1,border_width+x_height+1:2*border_width+x_height+1,i]=border_mat*image[y_height,x_height,i]
        # Setting the inner values equal to original image
        out_image[border_width:border_width+y_height+1,border_width:border_width+x_height+1,i]=image[:,:,i]
        # Copying and extending the values of the outer rows and columns of the original image
        out_image[:border_width,border_width:border_width+x_height+1,i]= np.tile(image[0,:,i],(border_width,1))
        out_image[border_width+y_height+1:2*border_width+y_height+1,border_width:border_width+x_height+1,i] = np.tile(image[y_height,:,i],(border_width,1))
        out_image[border_width:border_width+y_height+1,:border_width,i]=np.transpose(np.tile(image[:,0,i],(border_width,1)))
        out_image[border_width:border_width+y_height+1,border_width+x_height+1:2*border_width+x_height+1,i]=np.transpose(np.tile(image[:,x_height,i],(border_width,1)))

    return out_image

test_matrix = np.random.normal(0,1,(10,10,1))
out_test = fill_border(test_matrix,3)
print(out_test)
print(out_test[:,-1])
"""