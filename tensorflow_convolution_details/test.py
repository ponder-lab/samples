#!/usr/bin/env python

import tensorflow as tf
import numpy as np

def extract_patches(img, kernel, stride, padding):

    batch = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]
    ich = img.shape[3]
    
    kernel_y = kernel[0]
    kernel_x = kernel[1]
    stride_y = stride[0]
    stride_x = stride[1]
    
    if padding == "VALID":
        osize = (int(np.ceil((height - kernel_y + 1) / stride_y)),
                 int(np.ceil((width - kernel_x + 1) / stride_x)))
        pad_y = pad_x = pad_t = pad_b = pad_l = pad_r = 0

    elif padding == "SAME":
        osize = (int(np.ceil(height / stride_y)),
                 int(np.ceil(width / stride_x)))

        height_rem = height % stride_y
        if height_rem == 0:
            pad_y = np.max([kernel_y - stride_y, 0])
        else:
            pad_y = np.max([kernel_y - height_rem, 0])

        width_rem = width % stride_x
        if width_rem == 0:
            pad_x = np.max([kernel_x - stride_x, 0])
        else:
            pad_x = np.max([kernel_x - width_rem, 0])

        pad_t = int(np.floor(pad_y / 2))
        pad_b = pad_y - pad_t

        pad_l = int(np.floor(pad_x / 2))
        pad_r = pad_x - pad_l

    else:
        assert False, "Invalid padding type"

    buffed_shape = (batch, height + pad_y, width + pad_x, ich)
    img_buf = np.zeros(buffed_shape, img.dtype)

    img_buf[:, pad_t:pad_t+height, pad_l:pad_l+width, :] = img.copy()

    #print(np.squeeze(img_buf))
    #print(pad_t, pad_b, pad_l, pad_r)

    obuf = np.empty((batch,) + osize + (kernel_y * kernel_x * ich,), dtype=img.dtype)

    for bi in range(batch):
        for oy in range(osize[0]):
            iy = oy * stride_y
            for ox in range(osize[1]):
                ix = ox * stride_x
                
                pidx = 0
                for yi in range(kernel_y):
                    for xi in range(kernel_x):
                        for ci in range(ich):
                            obuf[bi, oy, ox, pidx] = img_buf[bi, iy+yi, ix+xi, ci]
                            pidx += 1
    return obuf


for i in range(100):
    height = np.random.randint(8, 16)
    width = np.random.randint(8, 16)
    ich = np.random.randint(1, 4)

    kernel = tuple(np.random.randint(1, 6, size=(2,)))
    stride = tuple(np.random.randint(1, 6, size=(2,)))
    padding = "SAME" if np.random.randint(0, 2) == 1 else "VALID"

    img = (np.arange(height*width*ich) + 1).reshape(1, height, width, ich)
    patch = tf.image.extract_patches(img,
                                     (1,) + kernel + (1,),
                                     (1,) + stride + (1,),
                                     (1, 1, 1, 1),
                                     padding=padding).numpy()
    
    my_patch = extract_patches(img, kernel, stride, padding)

    if not np.all(patch == my_patch):
        print("Wrong!")
        print(np.squeeze(img))
        print(np.squeeze(patch))
        print(np.squeeze(my_patch))

# height = 3
# width = 8
# ich = 1
# 
# kernel = 3
# stride = 3
# #padding = "VALID"
# padding = "SAME"
# 
# img = (np.arange(height*width*ich) + 1).reshape(1, height, width, ich)
# patch = tf.image.extract_patches(img,
#                                  (1, kernel, kernel, 1),
#                                  (1, stride, stride, 1),
#                                  (1, 1, 1, 1),
#                                  padding=padding).numpy()
# 
# my_patch = extract_patches(img, (kernel, kernel), (stride, stride), padding)
# 
# print(np.squeeze(img))
# print(np.squeeze(patch))
# print(np.squeeze(my_patch))

# 
# if padding == "SAME":
#     osize = np.ceil(width / stride)
#     print("output_size =", osize)
# 
#     padding = np.max([kernel - stride, 0]) if width % stride == 0 else np.max([kernel - width%stride, 0])
#     print("padding =", padding)
# 
#     padding_left = np.floor(padding/2)
#     print("padding_left =", padding_left)
# 
# elif padding == "VALID":
#     osize = np.ceil((width - kernel + 1) / stride)
#     print("output_size =", osize)
