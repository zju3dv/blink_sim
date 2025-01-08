from numpy import isposinf
import torch
import torch.nn.functional as F
import math

def rescale_intensity_frame(new_frame):
    """Rescale intensity frames.

    make sure we get no zero time constants
    limit max time constant to ~1/10 of white intensity level
    """
    return (new_frame+20)/275.

def subtract_leak_current(base_log_frame,
                          leak_rate_hz,
                          delta_time,
                          pos_thres,
                          leak_jitter_fraction,
                          noise_rate_array):
    """Subtract leak current from base log frame."""

    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32,
        device=noise_rate_array.device)

    curr_leak_rate = \
        leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

    delta_leak = delta_time*curr_leak_rate*pos_thres  # this is a matrix

    # ideal model
    #  delta_leak = delta_time*leak_rate_hz*pos_thres  # this is a matrix

    return base_log_frame-delta_leak



def generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        num_iters,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob):
    """Generate shot noise.
    """
    # new shot noise generator, generate for the entire batch
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time/num_iters) * \
        ((shot_noise_inten_factor-1)*inten01+1)
    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob

    # for shot noise
    rand01 = torch.rand(
        size=[num_iters]+list(inten01.shape),
        dtype=torch.float32,
        device=inten01.device)  # draw samples

    # pre compute all the shot noise cords
    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample.unsqueeze(0))
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample.unsqueeze(0))

    return shot_on_cord, shot_off_cord


def generate_shot_noise_linear(
        shot_noise_rate_hz,
        delta_time,
        num_iters,
        shot_noise_inten_factor,
        inten01,
        thres_pre_prob,
        is_positive
        ):
    """Generate shot noise.

    """
    # new shot noise generator, generate for the entire batch
    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time/num_iters) * \
        ((shot_noise_inten_factor-1)*inten01+1)
    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    if is_positive:
        shot_prob_this_sample = 1 - shot_noise_factor*thres_pre_prob
    else:
        shot_prob_this_sample = shot_noise_factor*thres_pre_prob

    # for shot noise
    rand01 = torch.rand(
        size=[num_iters]+list(inten01.shape),
        dtype=torch.float32,
        device=inten01.device)  # draw samples

    # pre compute all the shot noise cords
    if is_positive:
        shot_cord = torch.gt(rand01, shot_prob_this_sample.unsqueeze(0))
    else:
        shot_cord = torch.lt(rand01, shot_prob_this_sample.unsqueeze(0))

    return shot_cord


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray
        the input linear value in range 0-255
    :param threshold: float threshold 0-255
        the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()


def low_pass_filter(
        log_new_frame,
        lp_log_frame0,
        lp_log_frame1,
        inten01,
        delta_time,
        cutoff_hz=0):
    """Compute intensity-dependent low-pass filter.
    # Arguments
        log_new_frame: new frame in lin-log representation.
        lp_log_frame0:
        lp_log_frame1:
        inten01:
        delta_time:
        cutoff_hz:
    # Returns
        new_lp_log_frame0
        new_lp_log_frame1
    """
    if cutoff_hz <= 0:
        # unchange
        return log_new_frame, log_new_frame

    # else low pass
    tau = 1/(math.pi*2*cutoff_hz)

    # make the update proportional to the local intensity
    # the more intensity, the shorter the time constant
    eps = inten01*(delta_time/tau)
    eps = torch.clamp(eps, max=1)  # keep filter stable

    # first internal state is updated
    new_lp_log_frame0 = (1-eps)*lp_log_frame0+eps*log_new_frame

    # then 2nd internal state (output) is updated from first
    # Note that observations show that one pole is nearly always dominant,
    # so the 2nd stage is just copy of first stage
    new_lp_log_frame1 = lp_log_frame0
    # (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 # was 2nd-order,
    # now 1st order.

    return new_lp_log_frame0, new_lp_log_frame1


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """Compute event map.
    Prepare positive and negative event frames that later will be used
    for generating events.
    """
    # extract positive and negative differences
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    # compute quantized number of ON and OFF events for each pixel
    pos_evts_frame = torch.div(
        pos_frame, pos_thres, rounding_mode="floor").type(torch.int32)
    neg_evts_frame = torch.div(
        neg_frame, neg_thres, rounding_mode="floor").type(torch.int32)

    #  max_events = max(pos_evts_frame.max(), neg_evts_frame.max())

    #  # boolean array (max_events, height, width)
    #  # positive events and negative
    #  pos_evts_cord = torch.arange(
    #      1, max_events+1, dtype=torch.int32,
    #      device=diff_frame.device).unsqueeze(-1).unsqueeze(-1).repeat(
    #          1, diff_frame.shape[0], diff_frame.shape[1])
    #  neg_evts_cord = pos_evts_cord.clone().detach()
    #
    #  # generate event cords
    #  pos_evts_cord_post = (pos_evts_cord >= pos_evts_frame.unsqueeze(0))
    #  neg_evts_cord_post = (neg_evts_cord >= neg_evts_frame.unsqueeze(0))

    return pos_evts_frame, neg_evts_frame
    #  return pos_evts_cord_post, neg_evts_cord_post, max_events


def tensor_linspace(start, end, steps=10):
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

if __name__ == '__main__':
    start = torch.Tensor([0, 0])
    ts_step = torch.Tensor([1/3, 1/4])
    num_steps = max(3, 4)
    end = start+ts_step*num_steps
    result = tensor_linspace(start, end, steps=num_steps+1)
    print(result)