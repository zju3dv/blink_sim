import numpy as np
import cv2
import tqdm
import random

# apply colormap for 1-D sequential data in range[0, 1]
def apply_colormap(data, colormap=cv2.COLORMAP_VIRIDIS):
    data = data.copy().astype(float)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.clip(data, 0, 1)
    data = (data * 255).astype(np.uint8)
    data = cv2.applyColorMap(data, colormap)
    return data


def visualize_trajectory(particle_track, rgb_imgs, resolution, track_info,
                         valid=None, mask=None, segment_length=5, skip=50):
    num_frame = len(rgb_imgs)
    height, width = resolution
    frame_list = []

    if mask is None:
        track_count = len(track_info)
        mask = np.ones([track_count, height, width], dtype=bool)
    if valid is None:
        particle_frame_count = len(particle_track)
        valid = np.ones([particle_frame_count, height, width], dtype=bool)

    colormap = np.array([[random.randint(0, 255) for _ in range(3)] for _ in range(len(track_info))])

    x = np.array(range(skip, width, skip))
    y = np.array(range(skip, height, skip))
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()

    traj_imgs = np.zeros((num_frame, height, width, 3), np.uint8)

    cum_index = 0
    print(track_info)
    for track_ct, (track_start, track_end) in enumerate(track_info):
        particle_data = particle_track[cum_index:cum_index+(track_end-track_start)]
        valid_data = valid[cum_index:cum_index+(track_end-track_start)]
        for i in range(1, track_end-track_start):
            jj = max(1, i-segment_length+1)
            for j in range(jj, i+1):
                for k in range(len(xv)):
                    u, v = int(xv[k]), int(yv[k])
                    if not mask[track_ct,v,u]:
                        continue
                    if not valid_data[j-1,v,u] or not valid_data[j,v,u]:
                        continue
                    cv2.line(traj_imgs[i+track_start], tuple(particle_data[j-1,v,u].astype(int)),
                                tuple(particle_data[j,v,u].astype(int)),
                                tuple(colormap[track_ct].tolist()), thickness=2)
        cum_index += track_end - track_start

    for i in range(num_frame):
        rgb = rgb_imgs[i]
        alpha = np.sum(traj_imgs[i], -1) > 0
        alpha = np.stack([alpha] * 3, -1)
        rgb = alpha * traj_imgs[i] + (1 - alpha) * rgb
        frame_list.append(rgb.astype(np.uint8))

    return frame_list


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    h = max(h, 2)
    w = max(w, 2)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    v = v.astype(int)
    u = u.astype(int)

    v[v >= h] = h - 1
    v[v < 0] = 0
    u[u >= w] = w - 1
    u[u < 0] = 0

    color = vis[v.astype(np.int32), u.astype(np.int32)]
    return color


def visualize_trajectory_with_pred_and_gt(particle_gt=None, particle_pred=None, rgb_imgs=None, valid=None,
                                          segment_length=5, skip=50, thickness=2, particle_vis=['gt', 'pred'], vis_mode='traj'):
    # vis_mode {'traj', 'point', 'error'}
    # rgb_imgs  S, H, W, 3
    # particle  S, H, W, 2
    # valid     S, H, W
    # S num_frame
    num_frame = len(rgb_imgs)
    height, width = rgb_imgs[0].shape[:2]
    frame_list = []

    if valid is None:
        valid = np.ones([num_frame, height, width], dtype=bool)

    x = np.array(range(skip, width, skip))
    y = np.array(range(skip, height, skip))
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.flatten(), yv.flatten()

    particle_gt[particle_gt > 2 * max(height, width)] = 2 * max(height, width)
    particle_gt[particle_gt < -2 * max(height, width)] = -2 * max(height, width)
    particle_pred[particle_pred > 2 * max(height, width)] = 2 * max(height, width)
    particle_pred[particle_pred < -2 * max(height, width)] = -2 * max(height, width)

    traj_imgs = np.zeros((num_frame, height, width, 3), np.uint8)

    index = list(range(segment_length))
    colormap_gt = apply_colormap(np.array(index) / segment_length, colormap=cv2.COLORMAP_WINTER)    # blue
    colormap_pred = apply_colormap(np.array(index) / segment_length, colormap=cv2.COLORMAP_AUTUMN)  # red

    coords = particle_gt[0,valid[0].astype(bool)]
    www = coords[:,0].max() - coords[:,0].min()
    hhh = coords[:,1].max() - coords[:,1].min()
    pt_color = kp_color(xv - coords[:,0].min(), yv - coords[:,1].min(), (int(hhh+1), int(www)))
    
    error_max = int(min(height, width) / 2)
    index_error = list(range(error_max))
    colormap_error = apply_colormap(np.array(index_error) / error_max, colormap=cv2.COLORMAP_JET)

    for i in range(0, num_frame):
        if vis_mode == 'point' or vis_mode == 'error':
            jj = i
        elif vis_mode == 'error':
            jj = max(1, i-segment_length+1)
        for j in range(jj, i+1):
            for k in range(len(xv)):
                u, v = int(xv[k]), int(yv[k])
                if (vis_mode == 'point' or vis_mode == 'error') and not valid[j,v,u]:
                    continue
                if vis_mode == 'traj' and (not valid[j-1,v,u] or not valid[j,v,u]):
                    continue
                if vis_mode == 'point':
                    if 'gt' in particle_vis:
                        cv2.circle(traj_imgs[i],
                                tuple(particle_gt[j,v,u].astype(int)), 2,
                                tuple(pt_color[k].tolist()), thickness=thickness)
                    if 'pred' in particle_vis:
                        cv2.circle(traj_imgs[i],
                                tuple(particle_pred[j,v,u].astype(int)), 2,
                                tuple(pt_color[k].tolist()), thickness=thickness)
                elif vis_mode == 'traj':
                    if 'gt' in particle_vis:
                        cv2.line(traj_imgs[i],
                                tuple(particle_gt[j-1,v,u].astype(int)),
                                tuple(particle_gt[j,v,u].astype(int)),
                                tuple(colormap_gt[j-jj][0].tolist()), thickness=thickness)
                    if 'pred' in particle_vis:
                        cv2.line(traj_imgs[i],
                                tuple(particle_pred[j-1,v,u].astype(int)),
                                tuple(particle_pred[j,v,u].astype(int)),
                                tuple(colormap_pred[j-jj][0].tolist()), thickness=thickness)
                elif vis_mode == 'error':
                    error = np.linalg.norm(particle_pred[j,v,u] - particle_gt[j,v,u])
                    error = min(int(error), error_max - 1)
                    cv2.line(traj_imgs[i],
                             tuple(particle_pred[j,v,u].astype(int)),
                             tuple(particle_gt[j,v,u].astype(int)),
                             tuple(colormap_error[error][0].tolist()), thickness=thickness)

    for i in range(num_frame):
        rgb = rgb_imgs[i]
        alpha = np.sum(traj_imgs[i], -1) > 0
        alpha = np.stack([alpha] * 3, -1)
        rgb = alpha * traj_imgs[i] + (1 - alpha) * rgb
        frame_list.append(rgb.astype(np.uint8))

    return frame_list
