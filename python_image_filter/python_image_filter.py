#!/usr/bin/env python

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_dft(img, fsize):
    rows, cols = img.shape
    frows, fcols = fsize

    img_pad = np.zeros((frows, fcols), dtype=img.dtype)
    img_pad[:rows, :cols] = img
    img_pad = np.roll(img_pad, -int(rows / 2), axis=0)
    img_pad = np.roll(img_pad, -int(cols / 2), axis=1)

    f = np.fft.fft2(img_pad)
    fshift = np.fft.fftshift(f)

    return fshift

def apply_idft(fshift, img_size):
    rows, cols = img_size

    ff = np.fft.ifftshift(fshift)
    imgf_pad = np.abs(np.fft.ifft2(ff))
    imgf_pad = np.roll(imgf_pad, int(rows / 2), axis=0)
    imgf_pad = np.roll(imgf_pad, int(cols / 2), axis=1)

    imgf = imgf_pad[:rows, :cols]

    return imgf

img = cv2.imread("../python_image_fft/253027.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = (img.shape[0] - img.shape[0] % 8, img.shape[1] - img.shape[1] % 8)
img = img[:rows, :cols]
img = img.astype(np.float32) / 255.0

kernel_size = 15
kernel_g = cv2.getGaussianKernel(kernel_size, kernel_size/4)
kernel_g = kernel_g * kernel_g.T

imgf = cv2.filter2D(img, cv2.CV_32F, kernel_g)

fsize = 512
fin = apply_dft(img, (fsize, fsize))
fkn = apply_dft(kernel_g, (fsize, fsize))

fout = fin * fkn

imgff = apply_idft(fout, (rows, cols))

plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray", vmin=0, vmax=1), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(imgf, cmap="gray", vmin=0, vmax=1), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(imgff, cmap="gray", vmin=0, vmax=1), plt.xticks([]), plt.yticks([])
plt.show()
