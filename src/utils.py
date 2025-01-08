import os
import cv2
import random
import numpy as np
def safe_sample(data, num):
    safe_data = data
    if len(data) <= num:
        safe_data = data * (num // len(data) + 1)
    return random.sample(safe_data, num)

def img2video(input_dir, output_dir, num_f, subfix='_colors', save_name='colors.mp4', duration = 0.1):
    ffmpeg_input = ""
    for i in range(num_f):
        p = f"{input_dir}/{i}{subfix}.png"
        ffmpeg_input += f'file {os.path.abspath(p)}\n'
        ffmpeg_input += f'duration {duration}\n'
    # ffmpeg_input += f'file {os.path.abspath(p)}'

    with open(f"{input_dir}/ffmpeg_input.txt", 'wt') as f:
        f.write(ffmpeg_input)

    os.system(f"ffmpeg -nostdin -f concat -safe 0 -i {input_dir}/ffmpeg_input.txt"
        f" -vsync vfr -pix_fmt yuv420p {output_dir}/{save_name}")



def clean_tmp_files(output_dir):
    # os.system(f'rm -r {output_dir}/frames')
    os.system(f'rm -r {output_dir}/hdf5')
    
def check_blender_result(output_dir):
    slow_num = len(os.listdir(f'{output_dir}/hdf5/slow'))
    fast_num = len(os.listdir(f'{output_dir}/hdf5/fast'))
    return slow_num > 0 and fast_num > 0

def clean_unfinished(output_dir):
    print(f'removing {output_dir}')
    os.system(f'rm -r {output_dir}')

def clean_unfinished_all(data_root):
    for scene_name in sorted(os.listdir(f'{data_root}')):
        for seq_name in (sorted(os.listdir(f'{data_root}/{scene_name}'))):
            flag1 = os.path.exists(f'{data_root}/{scene_name}/{seq_name}/events_voxel')
            flag2 = os.path.exists(f'{data_root}/{scene_name}/{seq_name}/events_left')
            if not flag1 and not flag2:
                print(f'removing {data_root}/{scene_name}/{seq_name}')
                os.system(f'rm -r {data_root}/{scene_name}/{seq_name}')

def make_video(images, outvid=None, fps=10, size=None, is_color=True, format="mp4v"):
    if len(images) < 1:
        return
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

    try:
        import bpy
        bpy.context.scene.last_seed = seed
    except:
        pass

if __name__ == '__main__':
    clean_unfinished_all('/mnt/nas_8/datasets/eflyingthings/train')