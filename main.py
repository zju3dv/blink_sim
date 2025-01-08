# python main.py

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from numpy import require
import yaml
import numpy as np
import random
from src.utils import clean_tmp_files, check_blender_result, clean_unfinished
from src.blender.launcher import blender_generate_images_v2
from src.blender.visHdf5Files import parse_hdf5_to_flow_dataset, parse_hdf5_to_img_video3
from src.video2event import make_events

def main(config):
    # num_frames = config['num_frame']
    rgb_fps = config['rgb_image_fps']
    event_fps = config['event_image_fps']
    duration = config['duration']
    seq_range = config['seq_range']
    train_ratio = config['train_split_ratio']
    size = (config['image_height'], config['image_width'])
   
    mode = 'train'
    save_dir = "output/"
    num_seq = seq_range[1] - seq_range[0]
    for i in range(seq_range[0], seq_range[1]):
        np.random.seed(i)
        random.seed(i)
        if i > num_seq * train_ratio+seq_range[0]:
            mode = 'test'
        output_dir = f'{save_dir}/{mode}/{i:06d}'
        blender_generate_images_v2(config_file, output_dir, mode)
        status = check_blender_result(output_dir)
        if not status:
            clean_unfinished(output_dir)
            continue
        parse_hdf5_to_img_video3(output_dir, 'fast', size, int(duration*event_fps))
        parse_hdf5_to_flow_dataset(output_dir, int(duration*rgb_fps), config['image_width'], config['image_height'])
        evt_np = make_events(output_dir, size, int(duration*event_fps), event_fps, True, False, num_bins=15)
        clean_tmp_files(output_dir)

        print(f'seq#{i} ok')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--seq_range', nargs="+", type=int, required=False)
    parser.add_argument('--config', type=str, required=False, default='configs/blinkflow_v1.yaml')
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if args.seq_range:
        config['seq_range'] = [args.seq_range[0], args.seq_range[1]]
    print('seq_range:', config['seq_range'])

    main(config)
