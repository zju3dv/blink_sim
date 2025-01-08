import os
import h5py
import glob
import numpy as np
import json
import cv2
from src.utils import make_video
from src.traj_viz import visualize_trajectory
from src.hdr2ldr import vis_hdr_image
from src.blender.flow_utils import flow_consistency_torch


def parse_hdf5_to_img_video3(output_dir, mode, size, num_frame):
    hdr_list, ldr_list, particle_img_list = [], [], []
    for i in range(0, num_frame):
        with h5py.File(f'{output_dir}/hdf5/{mode}/{i}.hdf5', 'r') as data:
            if 'clean' in data.keys():
                hdr_img = data['clean'][:]
                hdr_img = (vis_hdr_image(hdr_img)*255).astype(np.uint8)
                hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2BGR)
                hdr_list.append(hdr_img)

            if 'final' in data.keys():
                ldr_img = data['final'][:]
                ldr_img = (ldr_img*255).astype(np.uint8)
                ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_RGB2BGR)
                ldr_list.append(ldr_img)

    particle_track, particle_valid = [], []
    if os.path.exists(f'{output_dir}/particle.json'):
        with open(f'{output_dir}/particle.json', 'r') as f:
            track_info = json.load(f)
            for (track_start, track_end) in track_info:
                for track_frame in range(track_start, track_end):
                    file_name = f'{track_start:06d}_{track_frame:06d}.npz'
                    particle_data = np.load(f'{output_dir}/particle/{file_name}')['particle']
                    track = particle_data[...,:2] # (h, w, 2)
                    status = particle_data[...,3:4] # (h, w)
                    valid = status >= -1
                    particle_track.append(track)
                    particle_valid.append(valid)

    if len(particle_track) > 0 and len(hdr_list) > 0:
        particle_track = np.stack(particle_track, axis=0)
        particle_valid = np.stack(particle_valid, axis=0)
        particle_img_list = visualize_trajectory(particle_track, hdr_list,
                                                resolution=size,
                                                track_info=track_info,
                                                valid=particle_valid,
                                                segment_length=1,
                                                skip=50
                                                )

    make_video(hdr_list, f'{output_dir}/clean.mp4', fps=5)
    make_video(ldr_list, f'{output_dir}/final.mp4', fps=5)
    make_video(particle_img_list, f'{output_dir}/particle.mp4', fps=5)


def parse_hdf5_to_dataset(output_dir, nFrames, config):
    num = len(glob.glob(f"{output_dir}/hdf5/slow/*.hdf5"))
    assert nFrames <= num

    for key in ['clean', 'final', 'forward_flow', 'backward_flow',
                'depth', 'normals', 'instance_segmaps', 'occlusion_map',
                'clean_uint8', 'final_uint8']:
        if config[f'parse_{key}']:
            os.system(f'mkdir -p {output_dir}/{key}')

    if config.get(f'parse_stereo', False):
        for key in ['clean', 'final']:
            if config[f'parse_{key}']:
                os.system(f'mkdir -p {output_dir}/{key}_right')
        for key in ['clean_uint8', 'final_uint8']:
            if config[f'parse_{key}']:
                os.system(f'mkdir -p {output_dir}/{key}_right')

    for i in range(nFrames):
        hdf5_path = f"{output_dir}/hdf5/slow/{i}.hdf5"
        data = h5py.File(hdf5_path, 'r')

        for key in ['forward_flow', 'backward_flow', 'clean', 'final', 'depth', 'normals', 'instance_segmaps']:
            if config[f'parse_{key}']:
                if key not in data.keys():
                    continue
                key_data = data[key][:].astype(np.float32)
                np.savez_compressed(f'{output_dir}/{key}/{i:06d}', **{key: key_data})
            
        if config.get(f'parse_stereo', False):
            postfix = '_right'
            for key in ['clean', 'final']:
                if config[f'parse_{key}']:
                    if f'{key}{postfix}' not in data.keys():
                        continue
                    key_data = data[f'{key}{postfix}'][:].astype(np.float32)
                    np.savez_compressed(f'{output_dir}/{key}{postfix}/{i:06d}', **{key: key_data})

        for camera_type in ['left', 'right']:
            if camera_type == 'right' and not config.get('parse_stereo', False):
                continue
            postfix = '' if camera_type == 'left' else '_right'

            if config[f'parse_clean_uint8'] and f'clean{postfix}' in data.keys():
                key_data = vis_hdr_image(data[f'clean{postfix}'][:])
                key_data = (key_data*255).astype(np.uint8)
                key_data = cv2.cvtColor(key_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/clean_uint8{postfix}/{i:06d}.png', key_data)

            if config[f'parse_final_uint8'] and f'final{postfix}' in data.keys():
                key_data = data[f'final{postfix}'][:]
                key_data = (key_data*255).astype(np.uint8)
                key_data = cv2.cvtColor(key_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/final_uint8{postfix}/{i:06d}.png', key_data)

        if config['parse_occlusion_map']:
            if 'forward_flow' not in data.keys() or 'backward_flow' not in data.keys():
                continue
            forward = data['forward_flow'][:] # (h, w, 2)
            if i != nFrames-1:
                hdf5_path_next = f"{output_dir}/hdf5/slow/{i+1}.hdf5"
                data_next = h5py.File(hdf5_path_next, 'r')
                backward = data_next['backward_flow'][:] # (h, w, 2)
                valid = flow_consistency_torch(forward, backward).astype(np.float32)
                np.savez_compressed(f'{output_dir}/occlusion_map/{i:06d}', occlusion_map=valid)


def parse_blinkvision(output_dir, config):
    rgb_path = f'{output_dir}/hdf5/rgb/'
    gt_path = f'{output_dir}/hdf5/gt/'

    with open(f'{output_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
        nFrames = metadata['num_frames']
        unit_scale = metadata['unit_scale']

    for key in ['clean', 'final', 'forward_flow', 'backward_flow',
                'depth', 'normals', 'instance_segmaps', 'occlusion_map',
                'clean_uint8', 'final_uint8']:
        if config[f'parse_{key}']:
            os.system(f'mkdir -p {output_dir}/{key}')

    require_gt = False
    require_rgb = False
    for key in ['forward_flow', 'backward_flow', 'depth', 'normals', 'instance_segmaps', 'occlusion_map']:
        if config[f'parse_{key}']:
            require_gt = True
            break
    for key in ['clean', 'final', 'clean_uint8', 'final_uint8']:
        if config[f'parse_{key}']:
            require_rgb = True
            break

    if require_rgb and not os.path.exists(rgb_path):
        print(f'require rgb but rgb h5 do not exist')
        return

    if require_gt and not os.path.exists(gt_path):
        print(f'require gt but gt h5 do not exist')
        return

    if config.get(f'parse_stereo', False):
        for key in ['clean', 'final']:
            if config[f'parse_{key}']:
                os.system(f'mkdir -p {output_dir}/{key}_right')
        for key in ['clean_uint8', 'final_uint8']:
            if config[f'parse_{key}']:
                os.system(f'mkdir -p {output_dir}/{key}_right')

    for i in range(nFrames):
        if not require_rgb:
            break
        for camera_type in ['left', 'right']:
            if camera_type == 'left' and not config['parse_left']:
                continue
            if camera_type == 'right' and not config.get('parse_stereo', False):
                continue
            postfix = '' if camera_type == 'left' else '_right'

            rgb_path = f'{output_dir}/hdf5/rgb{postfix}/'
            hdf5_path = f"{rgb_path}/{i}.hdf5"
            data = h5py.File(hdf5_path, 'r')

            for key in ['clean', 'final']:
                if config[f'parse_{key}']:
                    if key not in data.keys():
                        continue
                    key_data = data[key][:].astype(np.float32)
                    np.savez_compressed(f'{output_dir}/{key}{postfix}/{i:06d}', **{key: key_data})
            
            if config[f'parse_clean_uint8'] and 'clean' in data.keys():
                key_data = vis_hdr_image(data['clean'][:])
                key_data = (key_data*255).astype(np.uint8)
                key_data = cv2.cvtColor(key_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/clean_uint8{postfix}/{i:06d}.png', key_data)

            if config[f'parse_final_uint8'] and 'final' in data.keys():
                key_data = data['final'][:]
                key_data = (key_data*255).astype(np.uint8)
                key_data = cv2.cvtColor(key_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/final_uint8{postfix}/{i:06d}.png', key_data)

    for i in range(nFrames):
        if not require_gt:
            break
        hdf5_path = f"{gt_path}/{i}.hdf5"
        data = h5py.File(hdf5_path, 'r')

        for key in ['forward_flow', 'backward_flow', 'depth', 'normals', 'instance_segmaps']:
            if config[f'parse_{key}']:
                if key not in data.keys():
                    continue
                key_data = data[key][:].astype(np.float32)
                if key == 'depth':
                    key_data = key_data * unit_scale
                np.savez_compressed(f'{output_dir}/{key}/{i:06d}', **{key: key_data})
 
        if config['parse_occlusion_map']:
            if 'forward_flow' not in data.keys() or 'backward_flow' not in data.keys():
                continue
            forward = data['forward_flow'][:] # (h, w, 2)
            if i != nFrames-1:
                hdf5_path_next = f"{gt_path}/{i+1}.hdf5"
                data_next = h5py.File(hdf5_path_next, 'r')
                backward = data_next['backward_flow'][:] # (h, w, 2)
                valid = flow_consistency_torch(forward, backward).astype(np.float32)
                np.savez_compressed(f'{output_dir}/occlusion_map/{i:06d}', occlusion_map=valid)
