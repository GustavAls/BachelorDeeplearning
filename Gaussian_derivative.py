import numpy as np
import cv2


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
