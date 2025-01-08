import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
import pickle
from tracemalloc import start
import cv2
import h5py
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np
from src.utils import img2video
from src.v2e.emulator import EventEmulator as v2e_simulator
from src.dvs_voltmeter.simulator import EventSim as voltmeter
from src.voxel_grid import VoxelGrid
try:
    from src.esim.build import esim
except:
    esim = None
    print("esim not found, skip")


def _ms_to_idx(ts_us):
    # us -> ms
    # import ipdb; ipdb.set_trace()
    ts_ms = ts_us / 1e3
    max_t = np.ceil(ts_ms[-1]).astype(np.int64)
    l = np.searchsorted(ts_ms, list(range(0, max_t)))
    return l

def _make_events_from_voltmeter(nFrames, input_dir, interpTimes):
    events_list = []
    emulator = voltmeter()

    for i in range(nFrames):
        frame = h5py.File(f'{input_dir}/{i}.hdf5', 'r')['hdr'][:] * 255
        frame = 0.3*frame[...,0] + 0.59*frame[...,1] + 0.11*frame[...,2]
        newEvents = emulator.generate_events(frame, interpTimes[i])
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
    except:
        import pdb; pdb.set_trace()
    return events_np


def _make_events_from_v2e(nFrames, input_dir, interpTimes):
    events_list = []
    emulator = v2e_simulator(device='cuda')

    for i in range(nFrames):
        frame = h5py.File(f'{input_dir}/{i}.hdf5', 'r')['hdr'][:] * 255
        frame = 0.3*frame[...,0] + 0.59*frame[...,1] + 0.11*frame[...,2]
        newEvents = emulator.generate_events(frame, interpTimes[i])
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
    except:
        import pdb; pdb.set_trace()
    return events_np

def _make_events_from_esim(nFrames, input_dir, interpTimes):
    events_list = []
    emulator = esim.EventSimulator()

    for i in range(nFrames):
        frame = h5py.File(f'{input_dir}/{i}.hdf5', 'r')['hdr'][:]
        frame = 0.3*frame[...,0] + 0.59*frame[...,1] + 0.11*frame[...,2]
        newEvents = emulator.generate_events(frame, int(interpTimes[i]*1e6))
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
        events_np[:, 0] /= 1e6
    except:
        import pdb; pdb.set_trace()
    return events_np



def make_events(output_dir, size, nFrames, fps=300, save_h5=False, save_event_voxel=False, delta_ms=100, num_bins=15):
    # input_dir = f"{output_dir}/frames"
    input_dir = f"{output_dir}/hdf5/fast/"

    interpTimes = np.linspace(0, nFrames/fps, nFrames, True).tolist()

    print(f'*** Stage 3/3: emulating DVS events from {nFrames} frames')
    events_np = _make_events_from_voltmeter(nFrames, input_dir, interpTimes)
    # events_np = _make_events_from_v2e(nFrames, input_dir, interpTimes)
    # events_np = _make_events_from_esim(nFrames, input_dir, interpTimes)
    # import ipdb; ipdb.set_trace()

    ts = events_np[:,0]*1e6
    ms_to_idx = _ms_to_idx(ts)
    if save_h5:
        # import ipdb; ipdb.set_trace()
        # np.save(f'{output_dir}/events.npy', events_np)
        os.system(f'mkdir -p {output_dir}/events_left')
        hf = h5py.File(f'{output_dir}/events_left/events.h5', 'w')
        events_np[events_np[:,3]<0, 3] = 0
        hf.create_dataset('events/t', data=ts.astype('u8'), compression="gzip", compression_opts=9)
        hf.create_dataset('events/y', data=events_np[:,1].astype('u2'), compression="gzip", compression_opts=9)
        hf.create_dataset('events/x', data=events_np[:,2].astype('u2'), compression="gzip", compression_opts=9)
        hf.create_dataset('events/p', data=events_np[:,3].astype('u1'), compression="gzip", compression_opts=9)
        # hf.create_dataset('t_offset', data=np.array(0).astype('i8'), compression="gzip", compression_opts=9)
        hf.create_dataset('ms_to_idx', data=ms_to_idx.astype('u8'), compression="gzip", compression_opts=9)
        hf.close()

        # we do not need to create rectify_map
        # hf = h5py.File(f'{output_dir}/events_left/rectify_map.h5', 'w')
        # x = np.linspace(0, size[1], size[1])
        # y = np.linspace(0, size[0], size[0])
        # xv, yv = np.meshgrid(x, y)
        # rectify_map = np.stack([xv, yv], axis=2)
        # hf.create_dataset('rectify_map', data=rectify_map.astype('<f4'), compression="gzip", compression_opts=9)
        # hf.close()

    if save_event_voxel:
        os.system(f'mkdir -p {output_dir}/events_voxel')
        names = ['event_volume_old', 'event_volume_new']
        delta_us = delta_ms * 1000
        duration_us = int(1/fps * 1e6 * nFrames)
        tolerance = int(1e-4 * 1e6)
        idx = 1
        events_np = torch.from_numpy(events_np).to(device='cuda')
        voxel_grid = VoxelGrid((num_bins, size[0], size[1]), normalize=True, device='cuda')
        voxel_list = []
        for tt in range(0, duration_us-tolerance, delta_us):
            ts_start = tt
            ts_end = tt + delta_us
            sidx = ms_to_idx[int(ts_start/1000)]
            eidx = ms_to_idx[min(len(ms_to_idx)-1, int(ts_end/1000))]
            t = events_np[sidx:eidx, 0]
            y = events_np[sidx:eidx, 1]
            x = events_np[sidx:eidx, 2]
            p = events_np[sidx:eidx, 3]

            event_representation = voxel_grid.convert({'p': p, 't': t, 'x': x, 'y': y})
            voxel_list.append(event_representation.cpu().numpy())
        
        num_event_frame = int(duration_us / delta_us)
        for i in range(1, num_event_frame):
            event_voxel = {}
            for j in range(len(names)):
                event_voxel[names[j]] = voxel_list[i+j-1]
            f = h5py.File(f'{output_dir}/events_voxel/{i:06d}.h5', 'w')
            f.create_dataset(names[0], data=event_voxel[names[0]], compression="gzip", compression_opts=9)
            f.create_dataset(names[1], data=event_voxel[names[1]], compression="gzip", compression_opts=9)
            f.close()
        torch.cuda.empty_cache()

    # view_events(output_dir, events_np, size, nFrames, fps)

    return events_np


def events_scatter_on_image(evt, size, value=1, bias_clip=False):
    # value = evt[:,3].tolist()
    evt_yx = tuple(evt[:, [2, 1]].astype(np.int32).transpose(1, 0).tolist())
    evt_img = np.zeros(size)
    np.add.at(evt_img, evt_yx, value)
    if bias_clip:
        # bias +100 , clip less that 255
        evt_img[evt_img!=0] += 100
        evt_img = evt_img.clip(0, 255).astype(np.uint8)
    return evt_img

def events_scatter_on_image_pos_vs_neg(evt, size):
    evt_im_pos = events_scatter_on_image(evt[evt[:,3]>=0], size, 1, True)
    evt_im_neg = events_scatter_on_image(evt[evt[:,3]<0], size, 1, True)
    evt_im_pvsn = np.stack((evt_im_pos, np.zeros_like(evt_im_pos), evt_im_neg), 2)
    return evt_im_pvsn


def view_events(output_dir, events_np, size, num_frames, fps):
    save_dir = f"{output_dir}/frames"
    for f_i in range(num_frames):
        t0, t1 = f_i / fps, (f_i+1) / fps
        evt_i = events_np[(events_np[:,0]>=t0) * (events_np[:,0]<t1)]
        evt_im_add = events_scatter_on_image(evt_i, size, evt_i[:,3].tolist())
        evt_im_abs = events_scatter_on_image(evt_i,size,1,True)[:,:,None].repeat(3,2)
        evt_im_pvsn = events_scatter_on_image_pos_vs_neg(evt_i, size)
        plt.imsave(f"{save_dir}/{f_i}_event_image_add.png", evt_im_add)
        plt.imsave(f"{save_dir}/{f_i}_event_image_abs.png", evt_im_abs)
        plt.imsave(f"{save_dir}/{f_i}_event_image_pos_vs_neg.png", evt_im_pvsn)
    evt_im_add = events_scatter_on_image(events_np, size, events_np[:,3].tolist())
    evt_im_abs = events_scatter_on_image(events_np,size,1,True)[:,:,None].repeat(3,2)
    evt_im_pvsn = events_scatter_on_image_pos_vs_neg(events_np, size)
    img2video(save_dir, output_dir, num_frames, '_event_image_pos_vs_neg', 'event_image_pos_vs_neg.mp4')


def test():
    output_dir = '/home/xianr/works/FlyingThings22/outputs/FlyingTv5_75'
    evt_np, evt_flow = make_events(output_dir, (512,512), 60)
    # view_events(output_dir, evt_np,(512,512), 40, 60)

if __name__=='__main__':
    test()
