import os
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["NUMEXPR_MAX_THREADS"]="6"
import h5py
import math
import numpy as np
from src.v2e.emulator import EventEmulator as v2e_simulator
from src.dvs_voltmeter.simulator import EventSim as voltmeter
import torch
import torch.nn.functional as F
try:
    from src.esim.build import esim
except:
    esim = None


def _ms_to_idx(ts_us, duration_ms):
    # us -> ms
    # import ipdb; ipdb.set_trace()
    ts_ms = ts_us / 1e3
    # max_t = np.ceil(ts_ms[-1]).astype(np.int64)
    # l = np.searchsorted(ts_ms, list(range(0, max_t)))
    l = np.searchsorted(ts_ms, list(range(0, duration_ms)))
    return l

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=-1).float()
    return coords[None].repeat(batch, 1, 1, 1)


def interpolate_images(first_image, second_image, forward_flow, backward_flow,
                       use_torch=False, torch_cuda=False, torch_dtype=torch.float64):
    H, W = first_image.shape

    coords = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack(coords, axis=-1)

    forward_particle = coords + forward_flow
    forward_particle[...,0] = np.clip(forward_particle[...,0], 0, W-1)
    forward_particle[...,1] = np.clip(forward_particle[...,1], 0, H-1)
    forward_flow = forward_particle - coords

    backward_particle = coords + backward_flow
    backward_particle[...,0] = np.clip(backward_particle[...,0], 0, W-1)
    backward_particle[...,1] = np.clip(backward_particle[...,1], 0, H-1)
    backward_flow = backward_particle - coords

    inter_num = math.ceil(np.max(np.abs(forward_flow))) + 1

    first_image = torch.from_numpy(first_image)[None, None].repeat(inter_num, 1, 1, 1).float().cuda()
    second_image = torch.from_numpy(second_image)[None, None].repeat(inter_num, 1, 1, 1).float().cuda()
    forward_flow = torch.from_numpy(forward_flow)[None].repeat(inter_num, 1, 1, 1).float().cuda()
    backward_flow = torch.from_numpy(backward_flow)[None].repeat(inter_num, 1, 1, 1).float().cuda()

    for i in range(inter_num):
        forward_flow[i] = (inter_num-i-1) / (inter_num-1) * forward_flow[i]
        backward_flow[i] = i / (inter_num-1) * backward_flow[i]
    
    coords = coords_grid(inter_num, H, W).float().contiguous().cuda()

    forward_grid = (forward_flow + coords).contiguous()
    forward_grid[:, :, :, 0] = (forward_grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    forward_grid[:, :, :, 1] = (forward_grid[:, :, :, 1] * 2 - H + 1) / (H - 1)
    warp_second = F.grid_sample(second_image, forward_grid, padding_mode='zeros', align_corners=True)

    backward_grid = (backward_flow + coords).contiguous()
    backward_grid[:, :, :, 0] = (backward_grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    backward_grid[:, :, :, 1] = (backward_grid[:, :, :, 1] * 2 - H + 1) / (H - 1)
    warp_first = F.grid_sample(first_image, backward_grid, padding_mode='zeros', align_corners=True)

    interpolated_list = []
    for i in range(inter_num):
        interpolated = (inter_num-i-1) / (inter_num-1) * warp_first[i] + i / (inter_num-1) * warp_second[i]
        interpolated = interpolated[0]
        if not use_torch:
            interpolated = interpolated.cpu().numpy()
        else:
            if not torch_cuda:
                interpolated = interpolated.cpu()
            if interpolated.dtype != torch_dtype:
                interpolated = interpolated.type(torch_dtype)

        interpolated_list.append(interpolated)

    return interpolated_list

# write a generator to read images
class ImageReader:
    def __init__(self, input_dir, nFrames, times, use_interpolation=False,
                    use_torch=False, torch_cuda=False, torch_dtype=torch.float64):
        self.input_dir = input_dir
        self.nFrames = nFrames
        self.use_interpolation = use_interpolation
        self.times = times
        self.use_torch = use_torch
        self.torch_cuda = torch_cuda
        self.torch_dtype = torch_dtype

    def __iter__(self):
        if not self.use_interpolation:
            for i in range(self.nFrames):
                frame = h5py.File(f'{self.input_dir}/{i}.hdf5', 'r')['images'][:] * 255
                frame = 0.3*frame[...,0] + 0.59*frame[...,1] + 0.11*frame[...,2]
                yield frame, self.times[i]
        else:
            for i in range(self.nFrames-1):
                first_h5 = h5py.File(f'{self.input_dir}/{i}.hdf5', 'r')
                second_h5 = h5py.File(f'{self.input_dir}/{i+1}.hdf5', 'r')
                first_image = first_h5['images'][:] * 255
                second_image = second_h5['images'][:] * 255
                first_image = 0.3*first_image[...,0] + 0.59*first_image[...,1] + 0.11*first_image[...,2]
                second_image = 0.3*second_image[...,0] + 0.59*second_image[...,1] + 0.11*second_image[...,2]
                forward_flow = first_h5['forward_flow'][:]
                backward_flow = second_h5['backward_flow'][:]
                image_list = interpolate_images(first_image, second_image, forward_flow, backward_flow,
                                                self.use_torch, self.torch_cuda, self.torch_dtype)
                ts_list = np.linspace(self.times[i], self.times[i+1], len(image_list))
                if i != self.nFrames-2:
                    image_list = image_list[:-1]
                for j in range(len(image_list)):
                    yield image_list[j], ts_list[j]


def _make_events_from_voltmeter(nFrames, input_dir, interpTimes, use_interpolation=False):
    events_list = []
    emulator = voltmeter(device='cuda')

    reader = ImageReader(input_dir, nFrames, interpTimes, use_interpolation=use_interpolation,
                            use_torch=True, torch_cuda=True, torch_dtype=torch.float64)
    for frame, frame_ts in reader:
        newEvents = emulator.generate_events(frame, frame_ts)
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
    except:
        events_np = None

    return events_np


def _make_events_from_v2e(nFrames, input_dir, interpTimes, use_interpolation=False):
    events_list = []
    emulator = v2e_simulator(device='cuda')

    reader = ImageReader(input_dir, nFrames, interpTimes, use_interpolation=use_interpolation)
    for frame, frame_ts in reader:
        newEvents = emulator.generate_events(frame, frame_ts)
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
    except:
        events_np = None

    return events_np

def _make_events_from_esim(nFrames, input_dir, interpTimes, use_interpolation=False):
    events_list = []
    emulator = esim.EventSimulator()

    reader = ImageReader(input_dir, nFrames, interpTimes, use_interpolation=use_interpolation)
    for frame, frame_ts in reader:
        newEvents = emulator.generate_events(frame / 255, frame_ts * 1e6)
        if newEvents is not None and newEvents.shape[0] > 0:
            events_list.append(newEvents)
    try:
        events_np = np.concatenate(events_list, axis=0)
        events_np[:, 0] /= 1e6
    except:
        events_np = None

    return events_np



def make_events(input_dir, output_dir, time_list, simulator_name='voltmeter', use_interpolation=False):
    # input_dir = f"{output_dir}/frames"
    # input_dir = f"{output_dir}/hdf5/fast/"

    nFrames = len(time_list)
    print(f'*** Stage 3/3: emulating DVS events from {nFrames} frames')
    assert simulator_name in ['voltmeter', 'v2e', 'esim']
    if simulator_name == 'esim':
        assert esim != None
    func = globals().get(f'_make_events_from_{simulator_name}')
    events_np = func(nFrames, input_dir, time_list, use_interpolation)

    if events_np is None:
        return False

    ts = events_np[:,0]*1e6
    ms_to_idx = _ms_to_idx(ts, duration_ms=int(time_list[-1]*1e3))

    os.system(f'mkdir -p {output_dir}')
    hf = h5py.File(f'{output_dir}/events.h5', 'w')
    events_np[events_np[:,3]<0, 3] = 0
    hf.create_dataset('events/t', data=ts.astype('u8'), compression="gzip", compression_opts=9)
    hf.create_dataset('events/y', data=events_np[:,1].astype('u2'), compression="gzip", compression_opts=9)
    hf.create_dataset('events/x', data=events_np[:,2].astype('u2'), compression="gzip", compression_opts=9)
    hf.create_dataset('events/p', data=events_np[:,3].astype('u1'), compression="gzip", compression_opts=9)
    # hf.create_dataset('t_offset', data=np.array(0).astype('i8'), compression="gzip", compression_opts=9)
    hf.create_dataset('ms_to_idx', data=ms_to_idx.astype('u8'), compression="gzip", compression_opts=9)
    hf.close()

    return True

def test():
    output_dir = '/home/xianr/works/FlyingThings22/outputs/FlyingTv5_75'
    evt_np, evt_flow = make_events(output_dir, (512,512), 60)
    # view_events(output_dir, evt_np,(512,512), 40, 60)

if __name__=='__main__':
    test()
