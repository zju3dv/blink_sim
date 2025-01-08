import random
import math
import numpy as np
import os
from typing import List
import scipy.interpolate as spi
import json

def ACESToneMapping(color, adapated_lum):
    # ref: https://zhuanlan.zhihu.com/p/21983679
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    color *= adapated_lum
    return (color * (A * color + B)) / (color * (C * color + D) + E)

def vis_hdr_image(color):
    is_list = False
    if isinstance(color, List):
        is_list = True
        color = np.stack(color, axis=0)

    # tone mapping
    color = ACESToneMapping(color, 0.5)

    # gamma correction
    color = np.power(color, 0.45)

    # quantization
    color = np.round(color * 255.0) / 255.0

    # clip
    color = np.clip(color, 0.0, 1.0)

    if is_list:
        color = [color[i] for i in range(color.shape[0])]

    return color

def hdr2ldr(color, config, exposure_time=None, mode='train'):
    # ref:
    # https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Liu_Single-Image_HDR_Reconstruction_CVPR_2020_supplemental.pdf
    # https://arxiv.org/pdf/2111.13679.pdf
    # 1. multipled by random exposure time
    # 2. clip to [0, 1]
    # 3. apply CRF samples from DORF dataset
    # 4. quantization to 8bit
    is_list = False
    if isinstance(color, List):
        is_list = True
        color = np.stack(color, axis=0)

    # step 1
    if exposure_time is not None and type(exposure_time) == float:
        color = color * exposure_time

    elif exposure_time is not None and type(exposure_time) == list:
        assert len(exposure_time) == 2
        log2_exposure_time = random.uniform(exposure_time[0], exposure_time[1])
        exposure_time = math.pow(2, log2_exposure_time)
        color = color * exposure_time
    
    elif exposure_time is None or type(exposure_time) == list:
        if mode == 'train':
            log2_exposure_time = random.uniform(-3.5, 3.5)
            exposure_time = math.pow(2, log2_exposure_time)
            color = color * exposure_time
        else:
            extreme_bright_ratio = 0
            extreme_dark_ratio = 0
            for frame in color:
                # RGB to gray
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
                extreme_bright_ratio += np.sum(frame > 240/255)
                extreme_dark_ratio += np.sum(frame < 15/255)

            if extreme_bright_ratio > extreme_dark_ratio:
                log2_exposure_time = random.choice(range(1, 4))
            else:
                log2_exposure_time = random.choice(range(-3, 0))
            exposure_time = math.pow(2, log2_exposure_time)
            for i in range(len(color)):
                color[i] = color[i] * exposure_time

    # step 2
    color = np.clip(color, 0.0, 1.0)

    # step 3
    with open('configs/dorf_curves.json', 'r') as fr:
        crf_list = json.load(fr)[mode]
        path = random.choice(crf_list)
    crf_curve = np.load(path)
    irradiance = crf_curve[:, 0]
    intensity = crf_curve[:, 1]
    interp_func = spi.interp1d(irradiance, intensity)
    ori_shape = color.shape
    color = interp_func(color.reshape([-1])).reshape(ori_shape)

    # step 4, quantization
    color = np.round(color * 255.0) / 255.0

    if is_list:
        color = [color[i] for i in range(color.shape[0])]

    return color