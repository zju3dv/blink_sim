#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   simulator.py
@Time    :   2022/7/12 22:42
@Author  :   Songnan Lin, Ye Ma
@Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
@Note    :   
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''

import os
import numpy as np
import torch
from .simulator_utils import event_generation

class EventSim(object):
    def __init__(
            self,
            output_folder: str = None, video_name: str = None, device='cpu'
    ):
        """
        Parameters
        ----------
        cfg: config
        output_folder: str
            folder of output data file
        video_name: str
            name of input video / output data file
        """

        # set parameters in model
        # self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 \
        #     = cfg.SENSOR.K[0], cfg.SENSOR.K[1], cfg.SENSOR.K[2], cfg.SENSOR.K[3], cfg.SENSOR.K[4], cfg.SENSOR.K[5]
        self.k1 = np.random.uniform(4, 6)
        self.k2 = np.random.uniform(18, 25)
        self.k3 = np.random.uniform(0.5e-4, 2.5e-4)
        self.k4 = 1e-7
        self.k5 = np.random.uniform(5e-8, 5e-9)
        self.k6 = 1e-5

        # output file
        # path = os.path.join(output_folder, video_name + '.npy')

        # init
        self.reset()
        self.device = device

    def reset(self):
        '''
            resets so that next use will reinitialize the base frame
        '''
        self.baseFrame = None
        self.t_previous = None  # time of previous frame

    def generate_events( self, new_frame, t_frame):
        """
        Notes:
            Compute events in new frame.
        Parameters
            new_frame: np.ndarray
                [height, width]
            t_frame: float32
                timestamp of new frame in second
        Returns
            events: np.ndarray if any events, else None
                [N, 4], each row contains [timestamp (us), y cordinate, x cordinate, sign of event].
        """

        if not isinstance(new_frame, torch.Tensor):
            new_frame = torch.from_numpy(new_frame).to(torch.float64)
        if new_frame.dtype != torch.float64:
            new_frame = new_frame.to(torch.float64)
        if new_frame.device != self.device:
            new_frame = new_frame.to(self.device)

        t_frame = float(t_frame)

        # ------------------
        # Initialization
        if self.baseFrame is None:
            self.baseFrame = new_frame
            self.t_previous = t_frame
            self.delta_vd_res = torch.zeros_like(new_frame)  # initialize residual voltage change $\Delta V_d^{res}$
            self.t_now = torch.ones_like(new_frame, dtype=torch.float) * self.t_previous
            self.thres_off = torch.ones_like(new_frame)
            self.thres_on = torch.ones_like(new_frame)
            return None

        if t_frame <= self.t_previous:
            raise ValueError("this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        # ------------------
        # Calculte distribution parameters of Brownian Motion with Drift in Eq. (10)(11)
        delta_light = (new_frame - self.baseFrame)  # delta L
        avg_light = (new_frame + self.baseFrame) / 2.0  # average L
        denominator = 1 / (avg_light + self.k2)
        mu_clean = (self.k1 * delta_light / (t_frame - self.t_previous)) * denominator
        mu = mu_clean + self.k4 + self.k5 * avg_light
        var_clean = (self.k3 * torch.sqrt(avg_light)) * denominator
        var = var_clean + self.k6
        ori_shape = mu.shape

        # ------------------
        # Event Generation!
        e_t, e_x, e_y, e_p, e_dvd = event_generation(self.thres_on, self.thres_off,
                                                     mu, var,
                                                     self.delta_vd_res, self.t_now, t_frame)
        if e_t.shape[0] > 0:
            e_p = 2 * e_p - 1
            event_tensor = torch.stack([e_t, e_y, e_x, e_p], dim=1)
            _, sorted_idx = torch.sort(e_t)
            event_tensor = event_tensor[sorted_idx, :]
            event_tensor = event_tensor.contiguous().cpu().numpy()
        else:
            event_tensor = None

        # Update
        self.delta_vd_res = e_dvd.reshape(ori_shape)
        self.t_now = torch.ones_like(self.t_now, device=self.t_now.device) * t_frame
        self.t_previous = t_frame
        self.baseFrame = new_frame

        return event_tensor