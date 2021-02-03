import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
plt.close('all')

def dilation33(image):
    # Makes a 3 by 3 dilation of the a 2D image, program crashes if not provided as such
    y_height, x_height = image.shape
    out_image = np.zeros(y_height,x_height,3)
    out_image[:,:,0] = np.vstack(image[1:],image[-1])
    out_image[:,:,1] = image
    out_image[:,:,2] = np.vstack(image[0],image[0:y_height-1])

    out_image2 = np.max(out_image, axis= 2)
    out_image[:,:,0] = np.concatenate([image[:,1:],image[:,-1]],axis = 1)
    out_image[:,:,1] = out_image2
    out_image[:,:,2] = np.concatenate([image[:,0],image[:,0:(x_height-1)]],axis = 1)
    out_image = np.max(out_image, axis=2)
    return out_image



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


def gaussian_derivative(image,sigma,i_order,j_order):
    # Calculates the Gaussian derivative of the i'th order and of the j'th order along the second axis

    maximum_sigma = float(3)
    filter_size = int(maximum_sigma*sigma+0.5)  # unclear as to the point of this
    image = fill_border(image,filter_size)
    x = np.asarray([i for i in range(-filter_size, filter_size+1)])
    gaussian_distribution = 1/(np.sqrt(2*np.pi)*sigma)*np.exp((x**2)/(-2*sigma**2))

    # first making the gaussian in convolution in the x direction
    if i_order== 0:
        gaussian = gaussian_distribution/np.sum(gaussian_distribution)
    elif i_order==1:
        gaussian = -(x/sigma**2)*gaussian_distribution
        gaussian = gaussian/(np.sum(x*gaussian))
    elif i_order==2:
        gaussian = (x**2/sigma**4-1/sigma**2)*gaussian_distribution
        gaussian = gaussian - sum(gaussian)/(len(x)) #shape of x may also be used but has only one dimension
        gaussian = gaussian/np.sum(0.5*x*x*gaussian)

    gaussian = np.vstack((gaussian, np.zeros((len(gaussian)-1,len(gaussian)))))
    out_image = signal.convolve2d(gaussian,image,mode='valid')
    # subsequently in the y direction
    if j_order==0:
        gaussian = gaussian_distribution / np.sum(gaussian_distribution)
    elif j_order==1:
        gaussian = -(x / sigma ** 2) * gaussian_distribution
        gaussian = gaussian / (np.sum(x * gaussian))
    elif j_order==2:
        gaussian = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian_distribution
        gaussian = gaussian - sum(gaussian) / (len(x))  # shape of x may also be used but has only one dimension
        gaussian = gaussian / np.sum(0.5 * x * x * gaussian)
    gaussian = np.transpose(np.vstack((gaussian, np.zeros((len(gaussian)-1,len(gaussian))))))
    out_image = signal.convolve2d(gaussian,out_image,mode = 'valid')
    return out_image

# test on normally distributed data
test_img = np.random.normal(0,1,[100,100])
plt.figure(0)
plt.imshow(test_img)

test_img = gaussian_derivative(test_img,2,0,2)
plt.figure(1)
plt.imshow(test_img)
plt.show()


def general_color_constancy(image, gaussian_differentiation=0,sigma = 1,minkowski_norm=0, mask_image=0,saturation_threshold=255):

    y_height, x_height, dimension = image.shape
    if mask_image==0: mask_image = np.zeros((y_height,x_height))
    mask_image2 = mask_image + (dilation33(np.max(image,axis=2))>=saturation_threshold).astype(int)
    mask_image2 = (mask_image2 == 0).astype(int)

    # TODO wright set border
    mask_image2 = set_border(mask_image2,sigma +1)

    out_image = image

    if gaussian_differentiation == 0:
        if sigma != 0:
            for i in range(3):
                image[:,:,i] = gaussian_derivative(image,sigma, 0, 0)
    elif gaussian_differentiation > 0:
        # TODO wright norm_derivative
        Rx, Gx,Bx = norm_derivative(image, sigma, gaussian_differentiation)
        image[:,:,0] = Rx
        Image[:,:,1] = Gx
        Image[:,:,2] = Bx

    image = np.abs(image)










