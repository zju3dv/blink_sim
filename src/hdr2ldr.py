import random
import math
import numpy as np
from typing import List

def ACESToneMapping(color, adapated_lum):
    # ref: https://zhuanlan.zhihu.com/p/21983679
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    color *= adapated_lum
    return (color * (A * color + B)) / (color * (C * color + D) + E)

def tone_mapping(color):
    color = ACESToneMapping(color, 0.5)
    color = np.power(color, 0.45)
    color = np.clip(color, 0.0, 1.0)
    return color

def hdr2ldr(color, exposure_time = -1):
    # ref:
    # https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Liu_Single-Image_HDR_Reconstruction_CVPR_2020_supplemental.pdf
    # https://arxiv.org/pdf/2111.13679.pdf
    # 1. multipled by random exposure time
    # 2. clip to [0, 1]
    # 3. apply CRF samples from DORF dataset
    flag = False
    if isinstance(color, List):
        flag = True
        color = np.stack(color, axis=0)

    if exposure_time < 0:
        log2_exposure_time = random.uniform(-8, 0)
        exposure_time = math.pow(2, log2_exposure_time)
    color = color * exposure_time
    color = np.clip(color, 0.0, 1.0)
    # TODO: replace gamma2.2 with samples from DORF in the next step
    color = np.power(color, 0.45)
    color = np.clip(color, 0.0, 1.0)

    if flag:
        color = [color[i] for i in range(color.shape[0])]

    return color