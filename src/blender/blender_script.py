import blenderproc as bproc
from blenderproc.python.utility.Utility import UndoAfterExecution
from blenderproc.python.types.EntityUtility import Entity
from blenderproc.python.utility.Utility import KeyFrame
from blenderproc.python.renderer import RendererUtility
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
import argparse
import os
import shutil
import numpy as np
import math
import json
import h5py

# automative add the file into sys.path
import bpy
import sys
import os

# we get blend file path
filepath = bpy.data.filepath

# we get the directory relative to the blend file path
dir = os.path.dirname(filepath)

# we append our path to blender modules path
# we use if not statement to do this one time only
if not dir in sys.path:
   sys.path.append(dir)

import yaml
from scipy.spatial.transform import Rotation as R

from src.blender.animation import animation
from src.blender.video_corr import compute_video_corr, particle_to_flow
from src.blender.utils import (set_faceID_uv, unset_faceID_uv, triangulate_objs, set_category_id, \
                                        set_barycentric_coord_uv, unset_barycentric_coord_uv,
                                        enable_uv_output, disable_output, remove_output_entry_by_key)
from src.blender.env import (setup_env, set_engines, disable_alpha, recover_alpha, \
                                   setup_settings_config, backup_keyframe, rescale_keyframe, restore_keyframe,
                                   switch_to_right_camera, switch_to_left_camera, set_defocus_params,
                                   setup_atmospheric_effects, remove_atmospheric_effects)
from src.hdr2ldr import hdr2ldr, vis_hdr_image
from src.utils import set_random_seed
from src.blender.flow_utils import flow_consistency_numpy




def custom_print(to_print):
    print('-' * 10 + to_print + '-' * 10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/event.yaml')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir
    config_file = args.config_file
    mode = args.mode
    seed = args.seed

    set_random_seed(seed)

    os.system(f'mkdir -p {output_dir}/hdf5/slow')
    os.system(f'mkdir -p {output_dir}/hdf5/fast')
    os.system(f'mkdir -p {output_dir}/tmp')

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config['range_for_dynamic_obj'] = np.array(config['range_for_dynamic_obj'] )
        except yaml.YAMLError as exc:
            print(exc)

    assert config['engine_in_rgb_pass'] in ['cycles', 'eevee']
    assert config['animation_mode'] in ['linear', 'cubic_spline']

    bproc.init()

    cycles_config_file = f'configs/default_blender_settings.yaml'
    if os.path.exists(cycles_config_file):
        with open(cycles_config_file, "r") as stream:
            try:
                settings_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    setup_settings_config(settings_config, settings_config['_settings_attrs_'])

    setup_info = setup_env(config, mode)
    setup_info['output_dir'] = output_dir

    obj_pose_frames, cam_pose_frames = animation(setup_info, {
        'animation_mode': config['animation_mode'],
        'num_frame': int(config['rgb_image_fps'] * config['duration'] + 1),
        'num_keyframes': config['num_keyframes'],
        'nonlinear_velocity': config['nonlinear_velocity'],
    })
    all_obj_list = setup_info['all_obj_list']
    rigid_tracked_obj_list = setup_info['dynamic_objs']
    deformable_obj_list = []
    cam_list = [Entity(bpy.context.scene.camera)]

    if config.get('debug', False):
        if 'render_frame_range' in config:
            manual_frame_start, manual_frame_end = config['render_frame_range']
            bpy.context.scene.frame_start = manual_frame_start
            bpy.context.scene.frame_end = manual_frame_end

    data = dict()

    ########### first pass, render low-fps RGB ########### 
    engines = config['engine_in_rgb_pass']
    samples = config['render_samples_in_rgb_pass']

    for camera_type in ['left', 'right']:
        if camera_type == 'right':
            if not config['render_stereo']:
                continue
            original_cam_pos = switch_to_right_camera(config['baseline_length'])

        postfix = '' if camera_type == 'left' else '_right'

        if config['render_clean']:
            bproc.renderer.set_output_format("OPEN_EXR", 16)

            set_engines(engines, samples, long_animation=True, cpu_threads=config['cpu_threads_in_cycles'])
            bpy.context.scene.render.use_motion_blur = False
            clean_data = bproc.renderer.render(f'{output_dir}/tmp')
            custom_print('render RGB (clean data) finished')
            hdr_img = clean_data['colors']
            data[f'clean{postfix}'] = hdr_img

        if config['render_final']:
            # exr format which allows linear colorspace
            bproc.renderer.set_output_format("OPEN_EXR", 16)
            remove_output_entry_by_key('colors')

            set_engines(engines, samples, long_animation=True, cpu_threads=config['cpu_threads_in_cycles'])

            # motion blur
            bpy.context.scene.render.use_motion_blur = True
            blur_length_range = config['motion_blur_length_range']
            blur_length = np.random.uniform(blur_length_range[0], blur_length_range[1])
            bproc.renderer.enable_motion_blur(motion_blur_length=blur_length)

            # defocus blur
            if config.get('enable_defocus', False):
                base_range = config['focus_distance_base_range']
                base = np.random.uniform(base_range[0], base_range[1])
                distance = math.pow(2, base)
                set_defocus_params(True, distance)

            # atmospheric effects
            setup_atmospheric_effects(config)

            final_data = bproc.renderer.render(f'{output_dir}/tmp')
            custom_print('render RGB (final data) finished')

            # undo the motion blur, defocus blur and atmospheric effects
            bpy.context.scene.render.use_motion_blur = False
            set_defocus_params(False)
            remove_atmospheric_effects()

            ldr_img = final_data['colors']
            if config['enable_ldr_simulation']:
                exposure_time_range = config['exposure_time_range']
                ldr_img = hdr2ldr(ldr_img, config, exposure_time_range, mode='train')
            else:
                ldr_img = vis_hdr_image(ldr_img)
            data[f'final{postfix}'] = ldr_img

        if camera_type == 'right':
            switch_to_left_camera(original_cam_pos)

    ########### second pass, render ground truth data such as optical-flow/particle/depth/seg ########### 
    if config['render_gt']:
        remove_output_entry_by_key('colors')
        # the output format is for RGB, thus do not affect the GT data
        # and if use OPEN_EXR, the bwd_flow will be wrong
        bproc.renderer.set_output_format("PNG")
        bpy.context.scene.render.use_motion_blur = False

        with UndoAfterExecution():
            RendererUtility.render_init()
            # the amount of samples must be one and there can not be any noise threshold
            RendererUtility.set_max_amount_of_samples(1)
            RendererUtility.set_noise_threshold(0)
            RendererUtility.set_denoiser(None)
            RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)

            # Noting that the material will be changed in `disable_alpha`
            backup_alpha_table, backup_alpha_mute = disable_alpha(all_obj_list)

            bproc.renderer.enable_segmentation_output(map_by=['instance', 'name'], output_dir=f'{output_dir}/tmp')
            bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=f'{output_dir}/tmp')
            bproc.renderer.enable_normals_output(output_dir=f'{output_dir}/tmp')
            label_data = bproc.renderer.render(f'{output_dir}/tmp')

            disable_output('depth')
            disable_output('segmentation')
            disable_output('normal')

            recover_alpha(all_obj_list, backup_alpha_table, backup_alpha_mute)

            label_data.update(bproc.renderer.render_optical_flow(f'{output_dir}/tmp', f'{output_dir}/tmp', 
                get_backward_flow=True, get_forward_flow=True, blender_image_coordinate_style=False))
            del label_data['colors']
            custom_print('render depth/seg/normal finished')

            data.update(label_data)

            width, height = config['image_width'], config['image_height']
            num_frame = bpy.context.scene.frame_end - bpy.context.scene.frame_start
            depth = data['depth']
            instance_segmaps = data['instance_segmaps']

            render_flow_in_custom_stride = config.get('render_flow_in_custom_stride', False)
            render_particle = config.get('render_particle', False)
            render_triangle_uv = (render_flow_in_custom_stride or render_particle) and len(deformable_obj_list) > 0
            if render_triangle_uv:
                triangulate_objs(all_obj_list)
                enable_uv_output(output_dir=f'{output_dir}/tmp')

                backup_table = set_barycentric_coord_uv(all_obj_list)
                bary_data = bproc.renderer.render(f'{output_dir}/tmp', load_keys={'colors', 'uv'}, output_key='colors')
                unset_barycentric_coord_uv(all_obj_list, backup_table)
                barycentric = bary_data['uv']

                backup_table = set_faceID_uv(all_obj_list)
                face_data = bproc.renderer.render(f'{output_dir}/tmp', load_keys={'colors', 'uv'}, output_key='colors')
                unset_faceID_uv(all_obj_list, backup_table)
                face_id = face_data['uv']

                for frame in range(len(face_id)):
                    face_id[frame] = face_id[frame][...,0]*255 + face_id[frame][...,1]

                disable_output('uv')
            else:
                barycentric = None
                face_id = None

            if render_flow_in_custom_stride:
                flow_ref_frame = config['flow_ref_frame']
                flow_target_frame = config['flow_target_frame']
                if flow_ref_frame[1] == -1:
                    flow_ref_frame[1] = num_frame
                flow_pair = [[], []]
                for ref_frame in range(flow_ref_frame[0], min(flow_ref_frame[1], num_frame), flow_ref_frame[2]):
                    for target_frame in flow_target_frame:
                        if ref_frame + target_frame >= num_frame:
                            continue
                        flow_pair[0].append(ref_frame)
                        flow_pair[1].append(ref_frame + target_frame)
                forward_status_list, forward_flow_list = compute_video_corr(flow_pair[0], flow_pair[1], depth, instance_segmaps,
                                obj_pose_frames, cam_pose_frames, width, height, num_frame,
                                rigid_tracked_obj_list, deformable_obj_list,
                                config['occlusion_thres'], config['deform_error_tolerant_factor'],
                                face_id=face_id, barycentric=barycentric)
                backward_status_list, backward_flow_list = compute_video_corr(flow_pair[1], flow_pair[0], depth, instance_segmaps,
                                obj_pose_frames, cam_pose_frames, width, height, num_frame,
                                rigid_tracked_obj_list, deformable_obj_list,
                                config['occlusion_thres'], config['deform_error_tolerant_factor'],
                                face_id=face_id, barycentric=barycentric)

                forward_flow_list = particle_to_flow(forward_flow_list)
                backward_flow_list = particle_to_flow(backward_flow_list)
                
                forward_path = f'{output_dir}/forward_flow_custom_stride'
                backward_path = f'{output_dir}/backward_flow_custom_stride'
                occlusion_path = f'{output_dir}/occlusion_map_custom_stride'
                os.system(f'mkdir -p {forward_path}')
                os.system(f'mkdir -p {backward_path}')
                os.system(f'mkdir -p {occlusion_path}')
                for flow_ct, (ref_frame, target_frame) in enumerate(zip(flow_pair[0], flow_pair[1])):
                    forward_flow, forward_status = forward_flow_list[flow_ct], forward_status_list[flow_ct]
                    backward_flow, backward_status = backward_flow_list[flow_ct], backward_status_list[flow_ct]
                    forward_flow = np.concatenate([forward_flow, (forward_status!=-2)[...,None]], axis=-1)
                    backward_flow = np.concatenate([backward_flow, (backward_status!=-2)[...,None]], axis=-1)
                    occlusion = flow_consistency_numpy(forward_flow[...,:2], backward_flow[...,:2]).astype(np.float32)
                    file_name = f'{ref_frame:06d}_{target_frame:06d}'
                    np.savez_compressed(f'{forward_path}/{file_name}', forward_flow=forward_flow)
                    np.savez_compressed(f'{backward_path}/{file_name}', backward_flow=backward_flow)
                    np.savez_compressed(f'{occlusion_path}/{file_name}', occlusion_map=occlusion)

            if render_particle:
                num_frame = len(data['depth'])

                ref_start, ref_end, ref_duration, ref_step = config['particle_ref_frame']
                if ref_end == -1: ref_end = num_frame
                if ref_duration == -1: ref_duration = num_frame
                if ref_step == -1: ref_step = num_frame

                particle_pair = [[], []]
                for ref_frame in range(ref_start, ref_end, ref_step):
                    for target_frame in range(ref_duration):
                        if ref_frame + target_frame >= num_frame:
                            continue
                        particle_pair[0].append(ref_frame)
                        particle_pair[1].append(ref_frame + target_frame)
                particle_status_list, particle_track_list = compute_video_corr(particle_pair[0], particle_pair[1], depth, instance_segmaps,
                                obj_pose_frames, cam_pose_frames, width, height, num_frame,
                                rigid_tracked_obj_list, deformable_obj_list,
                                config['occlusion_thres'], config['deform_error_tolerant_factor'],
                                face_id=face_id, barycentric=barycentric)

                particle_path = f'{output_dir}/particle'
                os.system(f'mkdir -p {particle_path}')
                for particle_ct, (ref_frame, target_frame) in enumerate(zip(particle_pair[0], particle_pair[1])):
                    particle_status = particle_status_list[particle_ct]
                    particle_track = particle_track_list[particle_ct]
                    particle_data = np.concatenate([particle_track, particle_status[...,None]], axis=-1).astype(np.float32)
                    file_name = f'{ref_frame:06d}_{target_frame:06d}'
                    np.savez_compressed(f'{particle_path}/{file_name}', particle=particle_data)

                custom_print('render particle finished')

                with open(f'{output_dir}/particle.json', 'w') as f:
                    tracks =[]
                    for ref_frame in range(ref_start, ref_end, ref_step):
                        tracks.append([ref_frame, min(num_frame, ref_frame+ref_duration)])
                    json.dump(tracks, f, indent=4)

            set_category_id(all_obj_list)
            bproc.renderer.enable_segmentation_output(map_by=['category_id'], output_dir=f'{output_dir}/tmp', default_values={'category_id': 0})
            instance_data = bproc.renderer.render(f'{output_dir}/tmp')
            data['instance_segmaps'] = instance_data['category_id_segmaps']

            disable_output('segmentation')

        # we need the category id, so after th undo operation, set it again
        set_category_id(all_obj_list)

    if config['render_clean'] or config['render_final'] or config['render_gt']:
        bproc.writer.write_hdf5(f'{output_dir}/hdf5/slow', data)

        duration = config['duration']
        rgb_image_fps = config['rgb_image_fps']
        image_ts = np.linspace(0, duration, int(duration*rgb_image_fps+1)).tolist()
        with open(f'{output_dir}/image_ts.txt', 'w') as f:
            for ts in image_ts:
                f.write(f'{ts:.6f}\n')

    if config['render_gt']:
        instance_table = {}
        dynamic_instance_table = {}
        for obj in all_obj_list:
            if 'category_id' in obj.blender_obj and obj.blender_obj['category_id'] != 0:
                instance_table[obj.blender_obj['category_id']] = obj.blender_obj.name
        for obj in rigid_tracked_obj_list:
            dynamic_instance_table[obj.blender_obj['category_id']] = obj.blender_obj.name
        for obj in deformable_obj_list:
            if obj.blender_obj.type == 'MESH':
                dynamic_instance_table[obj.blender_obj['category_id']] = obj.blender_obj.name
                break
            for child in obj.blender_obj.children:
                if child.type == 'MESH':
                    dynamic_instance_table[child['category_id']] = child.name
                    break

        # save instance table in txt
        with open(f'{output_dir}/all_instance.txt', 'w') as f:
            for key, value in instance_table.items():
                f.write(f'{key} {value}\n')
        with open(f'{output_dir}/dynamic_instance.txt', 'w') as f:
            for key, value in dynamic_instance_table.items():
                f.write(f'{key} {value}\n')

        K_matrix = bproc.camera.get_intrinsics_as_K_matrix()

        with open(f'{output_dir}/metadata.json', 'w') as fw:
            json.dump({
                'rgb_fps': rgb_image_fps,
                'duration': duration,
                'K_matrix': K_matrix.tolist(),
            }, fw, indent=4)

    ########### third pass, render clean and high-fps image for event simulation ########### 
    if config['render_event']:
        bproc.renderer.set_output_format("OPEN_EXR", 16)
        bpy.context.scene.render.use_motion_blur = False
        remove_output_entry_by_key('colors')

        engines = config['engine_in_event_pass']
        samples = config['render_samples_in_event_pass']
        set_engines(engines, samples, long_animation=True, cpu_threads=config['cpu_threads_in_cycles'])

        rgb_image_fps = config['rgb_image_fps']
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end

        event_ts = [0]
        time_table = {frame_start: frame_start}
        acc_index = frame_start
        interval = 1 / rgb_image_fps

        backup_data = backup_keyframe(cam_list + rigid_tracked_obj_list + deformable_obj_list)

        for i, frame_index in enumerate(range(frame_start+1, frame_end)):
            if config['sample_mode'] == 'uniform':
                event_image_fps = config['event_image_fps']
                frame_num = math.ceil(event_image_fps / rgb_image_fps)
            elif config['sample_mode'] == 'adaptive':
                flow = data['forward_flow'][frame_index-1]
                H, W = flow.shape[:2]
                coords = np.meshgrid(np.arange(W), np.arange(H))
                coords = np.stack(coords, axis=-1)
                particle = coords + flow
                particle[...,0] = np.clip(particle[...,0], 0, W-1)
                particle[...,1] = np.clip(particle[...,1], 0, H-1)
                flow = particle - coords
                max_flow = np.max(np.abs(flow))
                frame_num = math.ceil(max_flow / config['max_pixel_movement'])
            time_table[frame_index] = frame_num + acc_index
            event_ts += [(i*interval + (j+1)*interval/frame_num) for j in range(frame_num)]
            acc_index += frame_num
        rescale_keyframe(cam_list + rigid_tracked_obj_list + deformable_obj_list, time_table=time_table)

        with open(f'{output_dir}/event_ts.txt', 'w') as f:
            for ts in event_ts:
                f.write(f'{ts:.6f}\n')

        event_data = bproc.renderer.render(f'{output_dir}/tmp')
        event_data['images'] = event_data.pop('colors')
        bproc.writer.write_hdf5(f'{output_dir}/hdf5/fast', event_data)

        restore_keyframe(cam_list + rigid_tracked_obj_list + deformable_obj_list, backup_data)

    shutil.rmtree(f'{output_dir}/tmp')


if __name__ == '__main__':
    main()





