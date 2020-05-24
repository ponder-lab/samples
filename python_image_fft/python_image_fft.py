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

frows, fcols = (512, 512)

img = cv2.imread("253027.jpg", cv2.IMREAD_GRAYSCALE)
img = img[:img.shape[0]-1, :img.shape[1]-1]

fshift = apply_dft(img, (frows, fcols))

frange = 64

# Rectangular HPF
fshift_h = fshift.copy()
fshift_h[frows//2-frange:frows//2+frange+1,
         fcols//2-frange:fcols//2+frange+1] = 0.0

imgf_h = apply_idft(fshift_h, img.shape)

# Rectangular LPF
fshift_l = fshift.copy()
fshift_l[:frows//2-frange, :] = 0.0
fshift_l[frows//2+frange+1:, :] = 0.0
fshift_l[:, :fcols//2-frange] = 0.0
fshift_l[:, fcols//2+frange+1:] = 0.0

imgf_l = apply_idft(fshift_l, img.shape)

# Gaussian LPF
fshift_g = fshift.copy()
gauss_ = cv2.getGaussianKernel(frows - 1, frange)
gauss_ = gauss_ * gauss_.T
gauss_ /= np.max(gauss_)
gauss = np.zeros((frows, fcols), dtype=np.float64)
gauss[1:, 1:] = gauss_
fshift_g *= gauss

imgf_g = apply_idft(fshift_g, img.shape)

vmin = np.min(np.log(np.abs(fshift)))
vmax = np.max(np.log(np.abs(fshift)))
plt.subplot(2, 4, 1), plt.imshow(img, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 5), plt.imshow(np.log(np.abs(fshift))), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(imgf_h, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(np.log(np.abs(fshift_h)), vmin=vmin, vmax=vmax), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.imshow(imgf_l, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(np.log(np.abs(fshift_l)), vmin=vmin, vmax=vmax), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(imgf_g, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(np.log(np.abs(fshift_g)), vmin=vmin, vmax=vmax), plt.xticks([]), plt.yticks([])
plt.show()

