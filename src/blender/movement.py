import blenderproc as bproc
import random, bpy
import numpy as np
from mathutils import Vector, Euler
import scipy.interpolate as spi
from scipy.spatial.transform import Rotation, RotationSpline, Slerp



def interpolate_a_frame(frame_idx, obj, location=None, rotation=None):
    if location is not None:
        obj.location = Vector(location)
        obj.keyframe_insert(data_path='location', frame=frame_idx)
    if rotation is not None:
        obj.rotation_euler = Euler(rotation, 'XYZ')
        obj.keyframe_insert(data_path='rotation_euler', frame=frame_idx)


def move_one_obj(obj, key_frames, locations=None, rotations=None, name='', mode='linear'):
    if locations is None: locations=[None]*len(key_frames)
    if rotations is None: rotations=[None]*len(key_frames)
    if not name: name = str(obj)[-13:-1]
    fs = np.linspace(key_frames[0], key_frames[-1], key_frames[-1]+1-key_frames[0], dtype=np.int32)
    if mode == 'linear':
        if locations[0] is None: locations_itp = None
        else:
            loc_x_linear = spi.interp1d(key_frames, [l[0] for l in locations])
            loc_y_linear = spi.interp1d(key_frames, [l[1] for l in locations])
            loc_z_linear = spi.interp1d(key_frames, [l[2] for l in locations])
            locations_itp = np.stack((loc_x_linear(fs), loc_y_linear(fs), loc_z_linear(fs))).transpose()
        if rotations[0] is None: rotations_itp = None
        else:
            rotations = Rotation.from_euler('XYZ', rotations, degrees=False)
            slerp = Slerp(key_frames, rotations)
            rotations_itp = slerp(fs).as_euler('XYZ', degrees=False)
    elif mode == 'cubinc_spline':
        if locations[0] is None: locations_itp = None
        else:
            # TODO: spline interpolate with seperate x,y,z is wrong?
            loc_x_cs = spi.CubicSpline(key_frames, [l[0] for l in locations])
            loc_y_cs = spi.CubicSpline(key_frames, [l[1] for l in locations])
            loc_z_cs = spi.CubicSpline(key_frames, [l[2] for l in locations])
            locations_itp = np.stack((loc_x_cs(fs), loc_y_cs(fs), loc_z_cs(fs))).transpose()
        if rotations[0] is None: rotations_itp = None
        else:
            rotations = Rotation.from_euler('XYZ', rotations, degrees=False)
            spline = RotationSpline(key_frames, rotations)
            rotations_itp = spline(fs).as_euler('XYZ', degrees=False)

    for k, f_i in enumerate(fs):
        loc = None if locations_itp is None else locations_itp[k]
        rot = None if rotations_itp is None else rotations_itp[k]
        interpolate_a_frame(f_i, obj, loc, rot)



def animation(output_dir, setup_info, config):
    mode = config['animation_mode']
    num_frames = config['num_frame']
    num_kf = config['num_keyframes']

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames

    if mode in ['linear', 'cubinc_spline']:
        key_frames = np.linspace(0, num_frames-1, num_kf).astype(np.int32).tolist()
        cam_pose_list = setup_info['cam_pose']
        obj_pose_list = setup_info['dyna_objs_pose']
        obj_list = setup_info['dynamic_objs']
        for obj_idx, _ in enumerate(obj_pose_list):
            obj_pos = [pos for (pos, euler) in obj_pose_list[obj_idx]]
            obj_euler = [euler for (pos, euler) in obj_pose_list[obj_idx]]
            obj = obj_list[obj_idx]
            move_one_obj(obj.blender_obj, key_frames, obj_pos, obj_euler, mode=mode)
        cam_pos = [pos for (pos, euler) in cam_pose_list]
        cam_euler = [euler for (pos, euler) in cam_pose_list]
        move_one_obj(bpy.context.scene.camera, key_frames, cam_pos, cam_euler, mode=mode)
    else:
        print('warning: we have deleted other modes')
        exit(-1)

