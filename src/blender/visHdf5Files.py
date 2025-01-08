import os
import h5py
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt
import sys
import json
import cv2
import re
import tqdm
import torch
import torch.nn.functional as F
from src.utils import img2video
from src.flow_viz import flow_to_image

default_rgb_keys = ["blur", "normals", "diffuse", "nocs"]
default_hdr_keys = ["colors"]
default_flow_keys = ["forward_flow", "backward_flow"]
default_segmap_keys = ["segmap", ".*_segmaps"]
default_segcolormap_keys = ["segcolormap"]
default_depth_keys = ["distance", "depth"]
all_default_keys = default_rgb_keys + default_flow_keys + \
    default_segmap_keys + default_segcolormap_keys + default_depth_keys
default_depth_max = 20


def key_matches(key, patterns, return_index=False):
    for p, pattern in enumerate(patterns):
        if re.fullmatch(pattern, key):
            return (True, p) if return_index else True

    return (False, None) if return_index else False


def vis_data(
        key, data, full_hdf5_data=None, file_label="", rgb_keys=None,
        hdr_keys=None, flow_keys=None, segmap_keys=None, segcolormap_keys=None,
        depth_keys=None, depth_max=default_depth_max, save_to_file=None):
    if rgb_keys is None:
        rgb_keys = default_rgb_keys[:]
    if hdr_keys is None:
        hdr_keys = default_hdr_keys[:]
    if flow_keys is None:
        flow_keys = default_flow_keys[:]
    if segmap_keys is None:
        segmap_keys = default_segmap_keys[:]
    if segcolormap_keys is None:
        segcolormap_keys = default_segcolormap_keys[:]
    if depth_keys is None:
        depth_keys = default_depth_keys[:]

    # If key is valid and does not contain segmentation data, create figure and add title
    if key_matches(key, flow_keys + rgb_keys + hdr_keys + depth_keys):
        plt.figure()
        plt.title("{} in {}".format(key, file_label))

    try:
        if key_matches(key, flow_keys):
            try:
                # This import here is ugly, but else everytime someone uses this script it demands opencv and the progressbar
                sys.path.append(os.path.join(os.path.dirname(__file__)))
                # from utils import flow_to_image
            except ImportError:
                raise ImportError(
                    "Using .hdf5 containers, which contain flow images needs opencv-python and progressbar "
                    "to be installed!")

            # Visualize optical flow
            if save_to_file is None:
                plt.imshow(flow_to_image(data))
            else:
                flow_data = flow_to_image(data)
                plt.imsave(save_to_file, flow_data)
                # try:
                #     plt.imsave(save_to_file, flow_to_image(data), cmap='jet')
                # except:
                #     import pdb; pdb.set_trace()
        elif key_matches(key, segmap_keys):
            # Try to find labels for each channel in the segcolormap
            channel_labels = {}
            _, key_index = key_matches(key, segmap_keys, return_index=True)
            if key_index < len(segcolormap_keys):
                # Check if segcolormap_key for the current segmap key is configured and exists
                segcolormap_key = segcolormap_keys[key_index]
                if full_hdf5_data is not None and segcolormap_key in full_hdf5_data:
                    # Extract segcolormap data
                    segcolormap = json.loads(
                        np.array(full_hdf5_data[segcolormap_key]).tostring())
                    if len(segcolormap) > 0:
                        # Go though all columns, we are looking for channel_* ones
                        for colormap_key, colormap_value in segcolormap[0].items():
                            if colormap_key.startswith("channel_") and colormap_value.isdigit():
                                channel_labels[int(
                                    colormap_value)] = colormap_key[len("channel_"):]

            # Make sure we have three dimensions
            if len(data.shape) == 2:
                data = data[:, :, None]
            # Go through all channels
            for i in range(data.shape[2]):
                # Try to determine label
                if i in channel_labels:
                    channel_label = channel_labels[i]
                else:
                    channel_label = i

                # Visualize channel
                if save_to_file is None:
                    plt.figure()
                    plt.title("{} / {} in {}".format(key,
                              channel_label, file_label))
                    plt.imshow(data[:, :, i], cmap='jet')
                else:
                    if data.shape[2] > 1:
                        filename = save_to_file.replace(
                            ".png", "_" + str(channel_label) + ".png")
                    else:
                        filename = save_to_file
                    plt.imsave(filename, data[:, :, i], cmap='jet')

        elif key_matches(key, depth_keys):
            # Make sure the data has only one channel, otherwise matplotlib will treat it as an rgb image
            if len(data.shape) == 3:
                if data.shape[2] != 1:
                    print(
                        "Warning: The data with key '" + key +
                        "' has more than one channel which would not allow using a jet color map. Therefore only the first channel is visualized.")
                data = data[:, :, 0]

            if save_to_file is None:
                im = plt.imshow(data, cmap='summer', vmax=depth_max)
                plt.colorbar()
            else:
                plt.imsave(save_to_file, data, cmap='summer', vmax=depth_max)
        elif key_matches(key, rgb_keys):
            if save_to_file is None:
                plt.imshow(data)
            else:
                data = np.clip(data, 0, 1)
                plt.imsave(save_to_file, data)
        elif key_matches(key, hdr_keys):
            import imageio
            if save_to_file is None:
                import pdb; pdb.set_trace()
                # plt.imshow(data)
            else:
                save_to_file = save_to_file.replace('png', 'exr')
                imageio.imwrite(save_to_file, data)
        else:
            if save_to_file is None:
                plt.imshow(data)
            else:
                plt.imsave(save_to_file, data)
        plt.close()
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        import pdb
        pdb.set_trace()


def vis_file(path, keys_to_visualize=None, rgb_keys=None, hdr_keys=None,
             flow_keys=None, segmap_keys=None, segcolormap_keys=None,
             depth_keys=None, depth_max=default_depth_max, save_to_path=None):
    if save_to_path is not None and not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    # Check if file exists
    if os.path.exists(path):
        if os.path.isfile(path):
            with h5py.File(path, 'r') as data:
                # print(path + ": ")

                # Select only a subset of keys if args.keys is given
                if keys_to_visualize is not None:
                    keys = [key for key in data.keys()
                            if key in keys_to_visualize]
                else:
                    keys = [key for key in data.keys()]

                # Visualize every key
                res = []
                for key in keys:
                    value = np.array(data[key])

                    if sum([ele for ele in value.shape]) < 5 or "version" in key:
                        if value.dtype == "|S5":
                            res.append((key, str(value).replace("[", "").replace(
                                "]", "").replace("b'", "").replace("'", "")))
                        else:
                            res.append((key, value))
                    else:
                        res.append((key, value.shape))

                if res:
                    res = ["'{}': {}".format(key, key_res)
                           for key, key_res in res]
                    # print("Keys: " + ', '.join(res))

                for key in keys:
                    value = np.array(data[key])
                    if save_to_path is not None:
                        save_to_file = os.path.join(
                            save_to_path, str(os.path.basename(path)).split('.')
                            [0] + "_" + key + ".png")
                    else:
                        save_to_file = None
                    # Check if it is a stereo image
                    if len(value.shape) >= 3 and value.shape[0] == 2:
                        # Visualize both eyes separately
                        for i, img in enumerate(value):
                            vis_data(
                                key, img, data, os.path.basename(path) +
                                (" (left)" if i == 0 else " (right)"),
                                rgb_keys, hdr_keys, flow_keys, segmap_keys,
                                segcolormap_keys, depth_keys, depth_max,
                                save_to_file)
                    else:
                        vis_data(
                            key, value, data, os.path.basename(path),
                            rgb_keys, hdr_keys, flow_keys, segmap_keys,
                            segcolormap_keys, depth_keys, depth_max,
                            save_to_file)
        else:
            print("The path is not a file")
    else:
        print("The file does not exist: {}".format(path))


def parse_hdf5_to_img_video3(output_dir, mode, size, num_frame):
    hdf5_paths = sorted(glob.glob(f"{output_dir}/hdf5/{mode}/*.hdf5"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'{output_dir}/hdr.mp4', fourcc, 10, (size[1], size[0]))
    for i in range(0, num_frame):
        with h5py.File(f'{output_dir}/hdf5/{mode}/{i}.hdf5', 'r') as data:
            img = data['hdr'][:]
            img = (np.clip(img, 0, 1)*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()

def parse_hdf5_to_img_video(output_dir):
    hdf5_paths = sorted(glob.glob(f"{output_dir}/hdf5/*.hdf5"))
    for hdf5_path in tqdm.tqdm(hdf5_paths):
        vis_file(
            path=hdf5_path,
            keys_to_visualize=["colors", "blur", "hdr", "forward_flow",
                               "backward_flow", "depth"],
            rgb_keys=["blur", "hdr", "normals", "diffuse", "nocs"],
            hdr_keys=["colors"],
            flow_keys=["forward_flow", "backward_flow"],
            segmap_keys=["segmap", ".*_segmaps", "instance_segmaps"],
            segcolormap_keys=["segcolormap"],
            depth_keys=["distance", "depth"],
            depth_max=80, save_to_path=f"{output_dir}/frames")

    num_f = len(hdf5_paths)
    dir_id = output_dir.split('_')[-1]
    in_dir = f"{output_dir}/frames"

    img2video(in_dir, output_dir, num_f, '_blur', 'blur.mp4')
    # img2video(in_dir, output_dir, num_f, '_depth', 'depth.mp4', dir_id)
    # img2video(in_dir, output_dir, num_f, '_normals', 'normals.mp4', dir_id)
    img2video(in_dir, output_dir, num_f, '_forward_flow', 'forward_flow.mp4')
    # img2video(in_dir, output_dir, num_f, '_backward_flow', 'backward_flow.mp4', dir_id)
    # img2video(in_dir, output_dir, num_f, '_instance_segmaps', 'instance_segmaps.mp4', dir_id)

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def flow_consistency(forward, backward):
    # Input:
    #     forward: HxWx2 numpy
    #     backward: HxWx2 numpy
    H, W, _ = forward.shape
    coords = coords_grid(1, H, W).cuda().float().contiguous()
    forward = torch.from_numpy(forward)[None].permute([0, 3, 1, 2]).cuda().float()
    backward = torch.from_numpy(backward)[None].permute([0, 3, 1, 2]).cuda().float()
    grid = (forward + coords).permute([0, 2, 3, 1]).contiguous()
    grid[:, :, :, 0] = (grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] * 2 - H + 1) / (H - 1)
    backward = F.grid_sample(backward, grid, padding_mode='zeros', align_corners=False)

    consistency = forward + backward
    consistency = consistency[0].permute([1,2,0])
    valid = torch.norm(consistency, dim=2) < 1
    valid = torch.unsqueeze(valid, dim=2).to(dtype=torch.float32, device='cpu').numpy()
    return valid


def parse_hdf5_to_flow_dataset(output_dir, nFrames, width, height):
    num = len(glob.glob(f"{output_dir}/hdf5/slow/*.hdf5"))
    assert nFrames <= num
    # os.system(f'mkdir -p {output_dir}/left_final')
    os.system(f'mkdir -p {output_dir}/hdr')
    os.system(f'mkdir -p {output_dir}/forward_flow')

    for i in range(nFrames):
        hdf5_path = f"{output_dir}/hdf5/slow/{i}.hdf5"
        data = h5py.File(hdf5_path, 'r')
        forward = data['forward_flow'][:] # (h, w, 2)

        hdr = data['blur'][:]
        np.save(f'{output_dir}/hdr/{i:06d}.npy', hdr)
        # blur = data['blur'][:] * 255
        # cv2.imwrite(f'{output_dir}/left_final/{i:06d}.png', blur)

        if i != nFrames-1:
            hdf5_path_next = f"{output_dir}/hdf5/slow/{i+1}.hdf5"
            data_next = h5py.File(hdf5_path_next, 'r')
            backward = data_next['backward_flow'][:] # (h, w, 2)
            valid = flow_consistency(forward, backward)

            flow_image = np.concatenate([forward, valid], axis=2)
            np.save(f'{output_dir}/forward_flow/{i:06d}.npy', flow_image)


def cli():
    parser = argparse.ArgumentParser("Script to visualize hdf5 files")

    parser.add_argument('hdf5_paths', nargs='+', help='Path to hdf5 file/s')
    parser.add_argument(
        '--keys', nargs='+',
        help='Keys that should be visualized. If none is given, all keys are visualized.',
        default=all_default_keys)
    parser.add_argument(
        '--rgb_keys', nargs='+',
        help='Keys that should be interpreted as rgb data.',
        default=default_rgb_keys)
    parser.add_argument(
        '--hdr_keys', nargs='+',
        help='Keys that should be interpreted as hdr data.',
        default=default_hdr_keys)
    parser.add_argument(
        '--flow_keys', nargs='+',
        help='Keys that should be interpreted as optical flow data.',
        default=default_flow_keys)
    parser.add_argument(
        '--segmap_keys', nargs='+',
        help='Keys that should be interpreted as segmentation data.',
        default=default_segmap_keys)
    parser.add_argument(
        '--segcolormap_keys', nargs='+',
        help='Keys that point to the segmentation color maps corresponding to the configured segmap_keys.',
        default=default_segcolormap_keys)
    parser.add_argument(
        '--depth_keys', nargs='+',
        help='Keys that contain additional non-RGB data which should be visualized using a jet color map.',
        default=default_depth_keys)
    parser.add_argument('--depth_max', type=float, default=default_depth_max)
    parser.add_argument('--save', default=None, type=str,
                        help='Saves visualizations to file.')

    args = parser.parse_args()

    # Visualize all given files
    for path in args.hdf5_paths:
        vis_file(
            path=path,
            keys_to_visualize=args.keys,
            rgb_keys=args.rgb_keys,
            hdr_keys=args.hdr_keys,
            flow_keys=args.flow_keys,
            segmap_keys=args.segmap_keys,
            segcolormap_keys=args.segcolormap_keys,
            depth_keys=args.depth_keys,
            depth_max=args.depth_max,
            save_to_path=args.save
        )
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    cli()
