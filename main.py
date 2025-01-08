import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
import yaml
import numpy as np
import random
import subprocess
from src.utils import clean_tmp_files, check_blender_result, clean_unfinished, set_random_seed
from src.blender.visHdf5Files import parse_hdf5_to_dataset, parse_hdf5_to_img_video3
from src.video2event import make_events

def blender_generate_images(config_file, output_dir, mode, seed):
    command = f'blenderproc run src/blender/blender_script.py' + \
              f' --config_file {config_file}' + \
              f' --output_dir {output_dir}' + \
              f' --seed {seed}' + \
              f' --mode {mode}'
    command_list = command.split(' ')

    p = subprocess.Popen(command_list)
    p.wait()


def main(config, config_file):
    # num_frames = config['num_frame']
    rgb_fps = config['rgb_image_fps']
    event_fps = config['event_image_fps']
    duration = config['duration']
    seq_range = config['seq_range']
    size = (config['image_height'], config['image_width'])
    mode = config['mode']
   
    save_dir = "output/"
    os.makedirs(f'{save_dir}/{mode}', exist_ok=True)

    with open(f'{save_dir}/{mode}/config.yaml', 'w') as f:
        ref_start, ref_end, ref_duration, ref_step = config['particle_ref_frame']
        num_rgb_frames= int(rgb_fps * duration + 1)
        if ref_end == -1: ref_end = num_rgb_frames
        if ref_duration == -1: ref_duration = num_rgb_frames
        if ref_step == -1: ref_step = num_rgb_frames
        data_config = {
            'fps': int(config['rgb_image_fps']),
            'duration': duration,
            'particle_ref_frame': [ref_start, ref_end, ref_duration, ref_step],
        }
        yaml.dump(data_config, f)

    seed_offset = 0 if mode == 'train' else 123456
    for i in range(seq_range[0], seq_range[1]):
        set_random_seed(i + seed_offset)
        output_dir = f'{save_dir}/{mode}/{i:06d}'
        try:
            blender_generate_images(config_file, output_dir, mode, i + seed_offset)
            status = check_blender_result(output_dir)
            if not status:
                # clean_unfinished(output_dir)
                continue
            parse_hdf5_to_img_video3(output_dir, 'slow', size, int(duration*rgb_fps+1))
            parse_hdf5_to_dataset(output_dir, int(duration*rgb_fps+1), config)
            if config.get('render_event', False):
                input_dir = f"{output_dir}/hdf5/fast/"
                num_event_frames = len(os.listdir(input_dir))
                if os.path.exists(f'{output_dir}/event_ts.txt'):
                    with open(f'{output_dir}/event_ts.txt', 'r') as f:
                        interpTimes = f.readlines()
                        interpTimes = [float(x.strip()) for x in interpTimes if x.strip()]
                else:
                    interpTimes = np.linspace(0, duration, num_event_frames, True).tolist()
                is_event_exist = make_events(input_dir, f'{output_dir}/events_left', time_list=interpTimes,
                                            simulator_name=config['event_simulator'])
                if is_event_exist:
                    clean_tmp_files(output_dir)
                    print(f'seq#{i} ok')
                else:
                    # clean_unfinished(output_dir)
                    print(f'delete seq#{i}, cause no events')
            else:
                clean_tmp_files(output_dir)
                print(f'seq#{i} ok')
        except Exception as e:
            import traceback
            print(f'seq#{i} error', e)
            traceback.print_exc()
            # write the error to log file, use appending
            with open(f'{mode}_error.log', 'a') as f:
                f.write('-'*8 + ' seq{i} failed ' + '-'*8 + '\n')
                f.write(f'{e}\n\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--seq_range', nargs="+", type=int, required=False)
    parser.add_argument('--config', type=str, required=False, default='configs/blinkflow_v2.yaml')
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

    main(config, config_file)
