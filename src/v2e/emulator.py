import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"

import math
import random

import numpy as np
import torch

from src.v2e.emulator_utils import (compute_event_map, generate_shot_noise,
                                      lin_log, low_pass_filter,
                                      rescale_intensity_frame,
                                      subtract_leak_current)


class EventEmulator(object):
    """compute events based on the input frame.
    - author: Zhe He
    - contact: zhehe@student.ethz.ch
    """

    def __init__(
            self,
            pos_thres=0.18,
            neg_thres=0.18,
            sigma_thres=0.03,
            cutoff_hz=300,
            leak_rate_hz=0.01,
            refractory_period_s=5*1e-4,
            shot_noise_rate_hz=1e-3,
            leak_jitter_fraction=0.1,
            noise_rate_cov_decades=0.1,
            seed=0,
            device="cuda"):
        """
        Parameters
        ----------
        base_frame: np.ndarray
            [height, width]. If None, then it is initialized from first data
        pos_thres: float, default 0.21
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        """

        self.base_log_frame = None
        self.t_previous = None  # time of previous frame

        # torch device
        self.device = device

        # thresholds
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres

        # non-idealities
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # event stats
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

    def _init(self, first_frame_linear):
        # base_frame are memorized lin_log pixel values
        self.base_log_frame = lin_log(first_frame_linear)

        # initialize first stage of 2nd order IIR to first input
        self.lp_log_frame0 = self.base_log_frame.clone().detach()
        # 2nd stage is initialized to same,
        # so diff will be zero for first frame
        self.lp_log_frame1 = self.base_log_frame.clone().detach()

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.pos_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)

            # to avoid the situation where the threshold is too small.
            self.pos_thres = torch.clamp(self.pos_thres, min=0.01)

            self.neg_thres = torch.normal(
                self.neg_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)
            self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

        # compute variable for shot-noise
        self.pos_thres_pre_prob = torch.div(
            self.pos_thres_nominal, self.pos_thres)
        self.neg_thres_pre_prob = torch.div(
            self.neg_thres_nominal, self.neg_thres)

        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of ON threshold below first frame value, to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        # do this *AFTER* we determine randomly distributed thresholds
        # (and use the actual pixel thresholds)
        # otherwise low threshold pixels will generate
        # a burst of events at the first frame
        if self.leak_rate_hz > 0:
            # no justification for this subtraction after having the
            # new leak rate model
            #  self.base_log_frame -= torch.rand(
            #      first_frame_linear.shape,
            #      dtype=torch.float32, device=self.device)*self.pos_thres

            # set noise rate array, it's a log-normal distribution
            self.noise_rate_array = torch.randn(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device)
            self.noise_rate_array = torch.exp(
                math.log(10)*self.noise_rate_cov_decades*self.noise_rate_array)

        # refractory period
        if self.refractory_period_s > 0:
            self.timestamp_mem = torch.zeros(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device)-self.refractory_period_s

    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.base_log_frame = None
        self.lp_log_frame0 = None  # lowpass stage 0
        self.lp_log_frame1 = None  # stage 1
        self.frame_counter = 0
        self.pos_thres = self.pos_thres_nominal
        self.neg_thres = self.neg_thres_nominal

    def generate_events(self, new_frame, t_frame):
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width]
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, y cordinate,
            x cordinate, sign of event].
            # TODO validate that this order of x and y is correctly documented
        """
        # update frame counter
        self.frame_counter += 1

        # convert into torch tensor
        new_frame = torch.tensor(new_frame, dtype=torch.float32,
                                 device=self.device)
        if self.base_log_frame is None:
            self._init(new_frame)
            self.t_previous = t_frame
            return None

        if t_frame <= self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        # lin-log mapping
        log_new_frame = lin_log(new_frame)

        # compute time difference between this and the previous frame
        delta_time = t_frame - self.t_previous

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # Time constant of the filter is proportional to
            # the intensity value (with offset to deal with DN=0)
            # limit max time constant to ~1/10 of white intensity level
            inten01 = rescale_intensity_frame(new_frame.clone().detach())

        # Apply nonlinear lowpass filter here.
        # Filter is a 1st order lowpass IIR (can be 2nd order)
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
        self.lp_log_frame0, self.lp_log_frame1 = low_pass_filter(
            log_new_frame=log_new_frame,
            lp_log_frame0=self.lp_log_frame0,
            lp_log_frame1=self.lp_log_frame1,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz)

        # Leak events: switch in diff change amp leaks at some rate
        # equivalent to some hz of ON events.
        # Actual leak rate depends on threshold for each pixel.
        # We want nominal rate leak_rate_Hz, so
        # R_l=(dI/dt)/Theta_on, so
        # R_l*Theta_on=dI/dt, so
        # dI=R_l*Theta_on*dt
        if self.leak_rate_hz > 0:
            self.base_log_frame = subtract_leak_current(
                base_log_frame=self.base_log_frame,
                leak_rate_hz=self.leak_rate_hz,
                delta_time=delta_time,
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array)

        # log intensity (brightness) change from memorized values is computed
        # from the difference between new input
        # (from lowpass of lin-log input) and the memorized value
        diff_frame = self.lp_log_frame1 - self.base_log_frame

        # generate event map
        pos_evts_frame, neg_evts_frame = compute_event_map(
            diff_frame, self.pos_thres, self.neg_thres)
        num_iters = max(pos_evts_frame.max(), neg_evts_frame.max())

        # record final events update
        final_pos_evts_frame = torch.zeros(
            pos_evts_frame.shape, dtype=torch.int32, device=self.device)
        final_neg_evts_frame = torch.zeros(
            neg_evts_frame.shape, dtype=torch.int32, device=self.device)
        # all events
        events = []

        ts_step = delta_time/num_iters
        ts = torch.linspace(
            start=self.t_previous+ts_step,
            end=t_frame,
            steps=num_iters, dtype=torch.float32, device=self.device)

        # NOISE: add temporal noise here by
        # simple Poisson process that has a base noise rate
        # self.shot_noise_rate_hz.
        # If there is such noise event,
        # then we output event from each such pixel

        # the shot noise rate varies with intensity:
        # for lowest intensity the rate rises to parameter.
        # the noise is reduced by factor
        # SHOT_NOISE_INTEN_FACTOR for brightest intensities

        # This was in the loop, here we calculate loop-independent quantities
        if self.shot_noise_rate_hz > 0:
            shot_on_cord, shot_off_cord = generate_shot_noise(
                    shot_noise_rate_hz=self.shot_noise_rate_hz,
                    delta_time=delta_time,
                    num_iters=num_iters,
                    shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                    inten01=inten01,
                    pos_thres_pre_prob=self.pos_thres_pre_prob,
                    neg_thres_pre_prob=self.neg_thres_pre_prob)

        for i in range(num_iters):
            # events for this iteration
            events_curr_iter = None

            # already have the number of events for each pixel in
            # pos_evts_frame, just find bool array of pixels with events in
            # this iteration of max # events

            # it must be >= because we need to make event for
            # each iteration up to total # events for that pixel
            pos_cord = (pos_evts_frame >= i+1)
            neg_cord = (neg_evts_frame >= i+1)

            # generate shot noise
            if self.shot_noise_rate_hz > 0:
                # update event list
                pos_cord = torch.logical_or(pos_cord, shot_on_cord[i])
                neg_cord = torch.logical_or(neg_cord, shot_off_cord[i])

            # filter events with refractory_period
            # only filter when refractory_period_s is large enough
            # otherwise, pass everything
            if self.refractory_period_s > ts_step:
                pos_time_since_last_spike = (
                    pos_cord*ts[i]-self.timestamp_mem)
                neg_time_since_last_spike = (
                    neg_cord*ts[i]-self.timestamp_mem)

                # filter the events
                pos_cord = (
                    pos_time_since_last_spike > self.refractory_period_s)
                neg_cord = (
                    neg_time_since_last_spike > self.refractory_period_s)

                # assign new history
                self.timestamp_mem = torch.where(
                    pos_cord, ts[i], self.timestamp_mem)
                self.timestamp_mem = torch.where(
                    neg_cord, ts[i], self.timestamp_mem)

            # update the base log frame, along with the shot noise
            final_pos_evts_frame += pos_cord
            final_neg_evts_frame += neg_cord

            # generate events
            # make a list of coordinates x,y addresses of events
            pos_event_xy = pos_cord.nonzero(as_tuple=True)
            neg_event_xy = neg_cord.nonzero(as_tuple=True)

            # update event stats
            num_pos_events = pos_event_xy[0].shape[0]
            num_neg_events = neg_event_xy[0].shape[0]
            num_events = num_pos_events + num_neg_events

            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            if num_events > 0:
                events_curr_iter = torch.ones(
                    (num_events, 4), dtype=torch.float32,
                    device=self.device)
                events_curr_iter[:, 0] *= ts[i]

                # pos_event cords
                events_curr_iter[:num_pos_events, 1] = pos_event_xy[0]
                events_curr_iter[:num_pos_events, 2] = pos_event_xy[1]

                # neg event cords
                events_curr_iter[num_pos_events:, 1] = neg_event_xy[0]
                events_curr_iter[num_pos_events:, 2] = neg_event_xy[1]

                # neg events polarity
                events_curr_iter[num_pos_events:, 3] *= -1

            # shuffle and append to the events collectors
            if events_curr_iter is not None:
                idx = torch.randperm(events_curr_iter.shape[0])
                events_curr_iter = events_curr_iter[idx].view(
                    events_curr_iter.size())
                events.append(events_curr_iter)

        # update base log frame according to the final
        # number of output events
        self.base_log_frame += final_pos_evts_frame*self.pos_thres
        self.base_log_frame -= final_neg_evts_frame*self.neg_thres

        if len(events) > 0:
            events = torch.vstack(events).cpu().data.numpy()

        # assign new time
        self.t_previous = t_frame
        if len(events) > 0:
            return events
        else:
            return None
