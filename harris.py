import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt
from skimage.io import imread


def gradient(kernel, imgray):
    return sig.convolve2d(imgray, kernel, mode='same')


def compute_spatial_derivative(imgray, kernel_x, kernel_y):
    return gradient(kernel_x, imgray), gradient(kernel_y, imgray)


def structure_tensor_setup(Ix, Iy):
    return ndi.gaussian_filter(Ix ** 2, sigma=1), ndi.gaussian_filter(Iy * Ix, sigma=1), ndi.gaussian_filter(Iy ** 2,
                                                                                                             sigma=1)


def harris_response(I_xx, I_xy, I_yy):
    k = 0.05
    det = (I_xx * I_yy) - (I_xy ** 2)
    trace = I_xx + I_yy

    return det - k * trace


def main():
    path = 'D:/Unitn/Progetti/harris_corner_detection/leaves.jpg'
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image = imread(path)
    print(image.shape)

    grayscale = rgb2gray(image)
    I_x, I_y = compute_spatial_derivative(grayscale, kernel_x, kernel_y)
    Ixx, Ixy, Iyy = structure_tensor_setup(I_x, I_y)
    response = harris_response(Ixx, Ixy, Iyy)
    img_copy = np.copy(image)
    for row_index, res in enumerate(response):
        for col_index, r in enumerate(res):

            if r > 0:
                img_copy[row_index, col_index] = [1, 0, 0]

    corners = corner_peaks(img_copy)
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '.r', markersize=3)
    plt.show()


main()
