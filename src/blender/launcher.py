import subprocess, signal, pickle
import numpy as np

def blender_generate_images_v2(config_file, output_dir, mode):
    command = f'blenderproc run src/blender/blender_script.py -config_file {config_file} -output_dir {output_dir} -mode {mode}'
    command_list = command.split(' ')

    p = subprocess.Popen(command_list)
    p.wait()


