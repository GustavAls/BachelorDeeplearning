import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy import signal
plt.close('all')

def dilation33(image):
    # Makes a 3 by 3 dilation of the a 2d image, program crashes if not provided as such
    y_height, x_height = image.shape
    out_image = np.zeros((y_height,x_height,3))

    out_image[:,:,0] = np.row_stack((image[1:,:],image[-1,:]))
    out_image[:,:,1] = image
    out_image[:,:,2] = np.row_stack((image[0,:],image[:(y_height-1),:]))

    out_image2 = np.max(out_image, axis= 2)
    out_image[:,:,0] = np.column_stack(([image[:,1:],image[:,-1]]))
    out_image[:,:,1] = out_image2
    out_image[:,:,2] = np.column_stack(([image[:,0],image[:,0:(x_height-1)]]))
    out_image = np.max(out_image, axis=2)
    return out_image
"""
test = np.random.normal(100,7, (50,50))
plt.figure(0)
plt.imshow(test)
test = dilation33(test)
plt.figure(1)
plt.imshow(test)
plt.show()
"""
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
        out_image[:border_width, :border_width] = border_mat * image[0, 0]
        out_image[border_width + y_height:2 * border_width + y_height, :border_width] = border_mat * image[y_height - 1, 0]
        out_image[:border_width, border_width + x_height:2 * border_width + x_height] = border_mat * image[0, x_height - 1]
        out_image[border_width + y_height:2 * border_width + y_height, border_width + x_height:2 * border_width + x_height] = border_mat * image[y_height - 1, x_height - 1]
        # Setting the inner values equal to original image
        out_image[border_width:border_width + y_height, border_width:border_width + x_height] = image[:, :]
        # Copying and extending the values of the outer rows and columns of the original image
        out_image[:border_width, border_width:border_width + x_height] = np.tile(image[0, :], (border_width, 1))
        out_image[border_width + y_height:2 * border_width + y_height, border_width:border_width + x_height] = np.tile(image[y_height - 1, :], (border_width, 1))
        out_image[border_width:border_width + y_height, :border_width] = np.transpose(np.tile(image[:, 0], (border_width, 1)))
        out_image[border_width:border_width + y_height, border_width + x_height:2 * border_width + x_height] = np.transpose(np.tile(image[:, x_height - 1], (border_width, 1)))
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


def gaussian_derivative(image, sigma, i_order, j_order):
    # Calculates the Gaussian derivative of the i'th order and of the j'th order along the second axis

    maximum_sigma = float(3)
    filter_size = int(maximum_sigma*sigma+0.5)  # unclear as to the point of this
    image = fill_border(image, filter_size)
    x = np.asarray([i for i in range(-filter_size, filter_size+1)])
    gaussian_distribution = 1/(np.sqrt(2*np.pi)*sigma)*np.exp((x**2)/(-2*sigma**2))
#   Gauss=1/(sqrt(2 * pi) * sigma)* exp((x.^2)/(-2 * sigma * sigma) );
    # first making the gaussian in convolution in the x direction
    if i_order == 0:
        gaussian = gaussian_distribution/np.sum(gaussian_distribution)
    elif i_order == 1:
        gaussian = -(x/sigma**2)*gaussian_distribution
        gaussian = gaussian/(np.sum(x*gaussian))
    elif i_order == 2:
        gaussian = (x**2/sigma**4-1/sigma**2)*gaussian_distribution
        gaussian = gaussian - sum(gaussian)/(len(x)) #shape of x may also be used but has only one dimension
        gaussian = gaussian/np.sum(0.5*x*x*gaussian)
    out_image = np.apply_along_axis(lambda m: signal.convolve(m, gaussian, mode='valid'), axis=1, arr=image)

    # subsequently in the y direction
    if j_order == 0:
        gaussian = gaussian_distribution / np.sum(gaussian_distribution)
    elif j_order == 1:
        gaussian = -(x / sigma ** 2) * gaussian_distribution
        gaussian = gaussian / (np.sum(x * gaussian))
    elif j_order == 2:
        gaussian = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian_distribution
        gaussian = gaussian - np.sum(gaussian) / (len(x))  # shape of x may also be used but has only one dimension
        gaussian = gaussian / np.sum(0.5 * x * x * gaussian)
    out_image = np.apply_along_axis(lambda m: signal.convolve(m, gaussian, mode='valid'), axis=0, arr=out_image)
    return out_image

# test on normally distributed data
"""
test_img = np.random.normal(0,1,[100,100])
plt.figure(0)
plt.imshow(test_img)

test_img = gaussian_derivative(test_img,2,0,2)
plt.figure(1)
plt.imshow(test_img)
plt.show()
"""
def norm_derivative(image, sigma, order = 1):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    if order == 1:
        Rx = gaussian_derivative(R, sigma, order, 0)
        Ry = gaussian_derivative(R, sigma, 0, order)
        Rw = np.sqrt(Rx ** 2 + Ry ** 2)

        Gx = gaussian_derivative(G, sigma, order, 0)
        Gy = gaussian_derivative(G, sigma, 0, order)
        Gw = np.sqrt(Gx ** 2 + Gy ** 2)

        Bx = gaussian_derivative(B, sigma, order, 0)
        By = gaussian_derivative(B, sigma, 0, order)
        Bw = np.sqrt(Bx ** 2 + By ** 2)

    elif order == 2:
        Rx = gaussian_derivative(R, sigma, order, 0)
        Ry = gaussian_derivative(R, sigma, 0, order)
        Rxy = gaussian_derivative(R, sigma, order // 2, order // 2)
        Rw = np.sqrt(Rx ** 2 + Ry ** 2 + 4 * Rxy ** 2)

        Gx = gaussian_derivative(G, sigma, order, 0)
        Gy = gaussian_derivative(G, sigma, 0, order)
        Gxy = gaussian_derivative(G, sigma, order // 2, order // 2)
        Gw = np.sqrt(Gx ** 2 + Gy ** 2 + 4 * Gxy ** 2)

        Bx = gaussian_derivative(B, sigma, order, 0)
        By = gaussian_derivative(B, sigma, 0, order)
        Bxy = gaussian_derivative(B, sigma, order // 2, order // 2)
        Bw = np.sqrt(Bx ** 2 + By ** 2 + 4 * Bxy ** 2)

    return Rw, Gw, Bw

def set_border(image, width, method = 0):
    y_height, x_height = image.shape
    temp = np.ones((y_height, x_height))
    y, x = np.meshgrid(np.arange(0, y_height), np.arange(0, x_height), indexing='ij')
    temp = temp * ((x < (x_height - width)) * (x > width))
    temp = temp * ((y < (y_height - width)) * (y > width))
    out = temp * image

    if method == 1:
        out = out + (np.sum(out) / np.sum(temp)) * (np.ones((y_height, x_height)) - temp)

    return out


def general_color_constancy(image, gaussian_differentiation=0, minkowski_norm=1, sigma=1, mask_image=0):

    y_height, x_height, dimension = image.shape
    if mask_image == 0:
        mask_image = np.zeros((y_height,x_height))

    #Removing saturated points
    saturation_threshold = 255
    mask_image2 = mask_image + (dilation33(np.max(image, axis=2)) >= saturation_threshold).astype(int)
    mask_image2 = (mask_image2 == 0).astype(int)

    mask_image2 = set_border(mask_image2, sigma + 1)

    out_image = np.copy(image)

    if gaussian_differentiation == 0:
        if sigma != 0:
            image = gaussian_derivative(image, sigma, 0, 0)
    elif gaussian_differentiation > 0:
        Rx, Gx, Bx = norm_derivative(image, sigma, gaussian_differentiation)
        image[:, :, 0] = Rx
        image[:, :, 1] = Gx
        image[:, :, 2] = Bx

    image = np.abs(image)

    if minkowski_norm != -1: #Minkowski norm = (1, infinity [
        kleur = np.power(image, minkowski_norm)
        white_R = np.power(np.sum(kleur[:, :, 0] * mask_image2), 1/minkowski_norm)
        white_G = np.power(np.sum(kleur[:, :, 1] * mask_image2), 1/minkowski_norm)
        white_B = np.power(np.sum(kleur[:, :, 2] * mask_image2), 1/minkowski_norm)

        som = np.sqrt(white_R ** 2 + white_G ** 2 + white_B ** 2)

        white_R = white_R / som
        white_G = white_G / som
        white_B = white_B / som

    else: #Minkowski norm is infinite, hence the max algorithm is applied
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        white_R = np.max(R * mask_image2)
        white_G = np.max(G * mask_image2)
        white_B = np.max(B * mask_image2)

        som = np.sqrt(white_R ** 2 + white_G ** 2 + white_B ** 2)

        white_R = white_R / som
        white_G = white_G / som
        white_B = white_B / som
    out_image[:, :, 0] = out_image[:, :, 0] / (white_R * np.sqrt(3))
    out_image[:, :, 1] = out_image[:, :, 1] / (white_G * np.sqrt(3))
    out_image[:, :, 2] = out_image[:, :, 2] / (white_B * np.sqrt(3))

    return white_R, white_G, white_B, out_image







#
# #test_img = np.random.normal(100, 20, size=(20, 20, 3))
# test_img = cv2.imread(r'C:\Users\Bruger\Pictures\melanomasTest.jpg', 1)
#
#
# plt.figure(0)
# #im = Image.fromarray(test_img.astype('uint8')).convert('RGB')
# im_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# plt.imshow(im_rgb)
#
#
# R, G, B, test_img1 = general_color_constancy(im_rgb, gaussian_differentiation=1, minkowski_norm=3, sigma=5)
# plt.figure(1)
# im1 = Image.fromarray(test_img1.astype('uint8')).convert('RGB')
# plt.imshow(im1)
#
# plt.show()




