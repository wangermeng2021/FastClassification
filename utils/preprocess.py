
import os
import logging
import time
import warnings
import numpy as np
import cv2
from PIL import Image


def normalize(img, mode=None):
    if mode == 'tf':
        img = img / 127.5 - 1.0
        # img = img / 255.
    elif mode == 'torch':  # tensorflow
        img /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img -= mean
        img /= std
    elif mode == 'caffe':  # caffe,bgr
        img=img.astype(np.float32)
        img -= [123.68, 116.779, 103.939]
    return img

def resize_img_aug(img, dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size / img_wh
    min_scale = np.min(scale)
    random_resize_style = np.random.randint(0, 5)
    resize_list = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale, interpolation=resize_list[random_resize_style])
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size // 2
    img = np.pad(img, [(half_pad_size[1], pad_size[1] - half_pad_size[1]),
                       (half_pad_size[0], pad_size[0] - half_pad_size[0]), (0, 0)],
                 constant_values=np.random.randint(0, 255))
    return img, min_scale, pad_size

def resize_img(img, dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size / img_wh
    min_scale = np.min(scale)
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale)
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size // 2
    img = np.pad(img, [(half_pad_size[1], pad_size[1] - half_pad_size[1]),
                       (half_pad_size[0], pad_size[0] - half_pad_size[0]), (0, 0)])
    return img, min_scale, pad_size

