import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy import ndimage
plt.close('all')

def dilation33(image):
    # Makes a 3 by 3 dilation of the a 2d image, program crashes if not provided as such
    y_height, x_height = image.shape
    out_image = np.zeros((y_height, x_height, 3))

    out_image[:, :, 0] = np.row_stack((image[1:, :], image[-1, :]))
    out_image[:, :, 1] = image
    out_image[:, :, 2] = np.row_stack((image[0, :], image[:(y_height-1), :]))

    out_image2 = np.max(out_image, axis=2)
    out_image[:, :, 0] = np.column_stack(([out_image2[:, 1:], out_image2[:, -1]]))
    out_image[:, :, 1] = out_image2
    out_image[:, :, 2] = np.column_stack(([out_image2[:, 0], out_image2[:, 0:(x_height-1)]]))
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



def gaussian_derivative(image, sigma, i_order, j_order,build_in = True):
    # Calculates the Gaussian derivative of the i'th order and of the j'th order along the second axis

    maximum_sigma = float(3)
    filter_size = int(maximum_sigma*sigma+0.5)  # unclear as to the point of this

    x = np.asarray([i for i in range(-filter_size, filter_size+1)])
    gaussian_distribution = 1/(np.sqrt(2*np.pi)*sigma)*np.exp((x**2)/(-2*sigma**2))
    # first making the gaussian in convolution in the x direction
    if not build_in:
        image = fill_border(image, filter_size)
        if i_order == 0:
            gaussian = gaussian_distribution/np.sum(gaussian_distribution)
        elif i_order == 1:
            gaussian = -(x/sigma**2)*gaussian_distribution
            gaussian = gaussian/(np.sum(x*gaussian))
        elif i_order == 2:
            gaussian = (x**2/sigma**4-1/sigma**2)*gaussian_distribution
            gaussian = gaussian - sum(gaussian)/(len(x)) #shape of x may also be used but has only one dimension
            gaussian = gaussian/np.sum(0.5*x*x*gaussian)
        gaussian = gaussian.reshape(gaussian.shape + (1,))
        out_image = signal.convolve2d(image, gaussian, mode='valid')


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
        gaussian = gaussian.reshape(gaussian.shape + (1,))
        out_image = signal.convolve2d(out_image, gaussian.T, mode='valid')

    else:
        if i_order == 0:
            out_image = ndimage.gaussian_filter1d(image, sigma, axis = 0,mode = 'reflect')
        if i_order == 1:
            out_image = ndimage.gaussian_filter1d(image,sigma, axis = 0, order = 1, mode = 'reflect')
        if i_order == 2:
            out_image = ndimage.gaussian_filter1d(image,sigma, axis = 0, order = 2, mode= 'reflect')

        if j_order == 0:
            out_image = ndimage.gaussian_filter1d(out_image, sigma, axis=1, mode='reflect')
        if j_order == 1:
            out_image = ndimage.gaussian_filter1d(out_image, sigma, axis=1, order=1, mode='reflect')
        if j_order == 2:
            out_image = ndimage.gaussian_filter1d(out_image, sigma, axis=1, order=2, mode='reflect')

    return out_image


def norm_derivative(image, sigma, order = 1, build_ind = True):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    if order == 1:
        Rx = gaussian_derivative(R, sigma, order, 0,build_ind)
        Ry = gaussian_derivative(R, sigma, 0, order,build_ind)
        Rw = np.sqrt(Rx ** 2 + Ry ** 2)
        Gx = gaussian_derivative(G, sigma, order, 0,build_ind)
        Gy = gaussian_derivative(G, sigma, 0, order,build_ind)
        Gw = np.sqrt(Gx ** 2 + Gy ** 2)

        Bx = gaussian_derivative(B, sigma, order, 0,build_ind)
        By = gaussian_derivative(B, sigma, 0, order,build_ind)
        Bw = np.sqrt(Bx ** 2 + By ** 2)

    elif order == 2:
        Rx = gaussian_derivative(R, sigma, order, 0,build_ind)
        Ry = gaussian_derivative(R, sigma, 0, order,build_ind)
        Rxy = gaussian_derivative(R, sigma, order // 2, order // 2,build_ind)
        Rw = np.sqrt(Rx ** 2 + Ry ** 2 + 4 * Rxy ** 2)

        Gx = gaussian_derivative(G, sigma, order, 0,build_ind)
        Gy = gaussian_derivative(G, sigma, 0, order,build_ind)
        Gxy = gaussian_derivative(G, sigma, order // 2, order // 2,build_ind)
        Gw = np.sqrt(Gx ** 2 + Gy ** 2 + 4 * Gxy ** 2)

        Bx = gaussian_derivative(B, sigma, order, 0,build_ind)
        By = gaussian_derivative(B, sigma, 0, order,build_ind)
        Bxy = gaussian_derivative(B, sigma, order // 2, order // 2,build_ind)
        Bw = np.sqrt(Bx ** 2 + By ** 2 + 4 * Bxy ** 2)

    return Rw, Gw, Bw


def set_border(image, width, method = 0):
    y_height, x_height = image.shape
    temp = np.ones((y_height, x_height))
    y, x = np.meshgrid(np.arange(1, y_height+1), np.arange(1, x_height+1), indexing='ij')
    temp = temp * ((x < (x_height - width + 1)) * (x > width))
    temp = temp * ((y < (y_height - width + 1)) * (y > width))
    out = temp * image

    if method == 1:
        out = out + (np.sum(out) / np.sum(temp)) * (np.ones((y_height, x_height)) - temp)

    return out



class general_color_constancy:

    def __init__(self,gaussian_differentiation=0, minkowski_norm=6, sigma=0, mask_image=0):
        self.gaussian_differentiation = gaussian_differentiation
        self.minkowski_norm = minkowski_norm
        self.sigma = sigma
        self.mask_image=mask_image

    def color_augment(self,image):
        # print("input image was of format {}".format(type(image)))
        image = np.array(image)
        # print("input image was changed to format {}".format(image.dtype))


        y_height, x_height, dimension = image.shape
        if self.mask_image == 0:
            mask_image = np.zeros((y_height, x_height))

    #Removing saturated points
    saturation_threshold = 255

    mask_image2 = mask_image + (dilation33(np.max(image, axis=2)) >= saturation_threshold)
    mask_image2 = (mask_image2 == 0)

    mask_image2 = set_border(mask_image2, sigma + 1)

    image_copy = np.ndarray.copy(image).astype(int)

    if gaussian_differentiation == 0:
        if sigma != 0:
            image_copy = gaussian_derivative(image_copy, sigma, 0, 0)
    elif gaussian_differentiation > 0:
        Rx, Gx, Bx = norm_derivative(image_copy, sigma, gaussian_differentiation, build_ind=False)
        image_copy[:, :, 0] = Rx
        image_copy[:, :, 1] = Gx
        image_copy[:, :, 2] = Bx

    image = np.fabs(image)

    if minkowski_norm != -1: #Minkowski norm = (1, infinity [
        kleur = np.float_power(image_copy, minkowski_norm)
        white_R = np.float_power(np.sum(kleur[:, :, 0] * mask_image2), (1/minkowski_norm))
        white_G = np.float_power(np.sum(kleur[:, :, 1] * mask_image2), (1/minkowski_norm))
        white_B = np.float_power(np.sum(kleur[:, :, 2] * mask_image2), (1/minkowski_norm))

        som = np.sqrt(white_R ** 2.0 + white_G ** 2.0 + white_B ** 2.0)

        white_R = white_R / som
        white_G = white_G / som
        white_B = white_B / som

    else: #Minkowski norm is infinite, hence the max algorithm is applied
        R = image_copy[:, :, 0]
        G = image_copy[:, :, 1]
        B = image_copy[:, :, 2]

        white_R = np.max(R * mask_image2)
        white_G = np.max(G * mask_image2)
        white_B = np.max(B * mask_image2)

        som = np.sqrt(white_R ** 2 + white_G ** 2 + white_B ** 2)

        white_R = white_R / som
        white_G = white_G / som
        white_B = white_B / som

    out_image = np.ndarray.copy(image).astype(int)
    out_image[:, :, 0] = image[:, :, 0] / (white_R * np.sqrt(3.0))
    out_image[:, :, 1] = image[:, :, 1] / (white_G * np.sqrt(3.0))
    out_image[:, :, 2] = image[:, :, 2] / (white_B * np.sqrt(3.0))

    #Makes sure there is no overflow
    out_image[out_image >= 255] = 255

    return white_R, white_G, white_B, out_image
#
# test_img = cv2.imread(r'C:\Users\Bruger\Pictures\building1.jpg', 1)
# # test_img = cv2.imread(r'C:\Users\ptrkm\OneDrive\Dokumenter\TestFolder\ISIC_0000001.jpg', 1)
# im_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# # imtest = np.random.normal(100,10, (250,250,3))
#
# R, G, B, test_img1 = general_color_constancy(im_rgb, gaussian_differentiation=1, minkowski_norm=5, sigma=2)
#
# fig = plt.figure(figsize=(9,12))
# fig.add_subplot(1,2,1)
# plt.imshow(im_rgb)
#
# fig.add_subplot(1,2,2)
# plt.imshow(test_img1)
#
# plt.show()
#




