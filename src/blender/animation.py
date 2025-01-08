import blenderproc as bproc
import random, bpy
import numpy as np
from mathutils import Vector, Euler
import scipy.interpolate as spi
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


def _apply_animation(frame_idx, obj, location=None, rotation=None):
    obj.location = Vector(location)
    obj.keyframe_insert(data_path='location', frame=frame_idx)

    obj.rotation_euler = Euler(rotation, 'XYZ')
    obj.keyframe_insert(data_path='rotation_euler', frame=frame_idx)

def interpolate_trajectory(num_frames, num_kf, locations, rotations, animation_mode='linear', nonlinear_velocity=False, spline_mode='cubic'):
    key_frames = np.linspace(0, num_frames-1, num_kf).astype(np.int32).tolist()
    fs = np.linspace(key_frames[0], key_frames[-1], key_frames[-1]+1-key_frames[0], dtype=np.int32)
    if nonlinear_velocity and num_kf > 1:
        kf_interval = key_frames[1] - key_frames[0]
        protect_interval = kf_interval / 4
        frame_idx = np.zeros(((num_kf - 1) * 2 + 1))
        frame_time = np.zeros(((num_kf - 1) * 2 + 1))
        for kf_idx in range(num_kf - 1):
            frame_idx[kf_idx * 2] = key_frames[kf_idx]
            frame_time[kf_idx * 2] = key_frames[kf_idx]
            frame_idx[kf_idx * 2 + 1] = (key_frames[kf_idx] + key_frames[kf_idx + 1]) / 2
            frame_time[kf_idx * 2 + 1] = np.random.uniform(key_frames[kf_idx] + protect_interval, key_frames[kf_idx + 1] - protect_interval)
        frame_idx[-1] = key_frames[-1]
        frame_time[-1] = key_frames[-1]
        # frame_time_nonlinear = spi.CubicSpline(frame_idx, frame_time)
        frame_time_nonlinear = spi.interp1d(frame_idx, frame_time, kind='quadratic')
        fs = frame_time_nonlinear(fs)
    if animation_mode == 'linear':
        loc_x_linear = spi.interp1d(key_frames, [l[0] for l in locations])
        loc_y_linear = spi.interp1d(key_frames, [l[1] for l in locations])
        loc_z_linear = spi.interp1d(key_frames, [l[2] for l in locations])
        locations_itp = np.stack((loc_x_linear(fs), loc_y_linear(fs), loc_z_linear(fs))).transpose()

        rot_x_linear = spi.interp1d(key_frames, [r[0] for r in rotations])
        rot_y_linear = spi.interp1d(key_frames, [r[1] for r in rotations])
        rot_z_linear = spi.interp1d(key_frames, [r[2] for r in rotations])
        rotations_itp = np.stack((rot_x_linear(fs), rot_y_linear(fs), rot_z_linear(fs))).transpose()

        # rotations = Rotation.from_euler('xyz', rotations, degrees=False)
        # slerp = Slerp(key_frames, rotations)
        # rotations_itp = slerp(fs).as_euler('xyz', degrees=False)

    elif animation_mode == 'cubic_spline':
        loc_x_cs = spi.CubicSpline(key_frames, [l[0] for l in locations])
        loc_y_cs = spi.CubicSpline(key_frames, [l[1] for l in locations])
        loc_z_cs = spi.CubicSpline(key_frames, [l[2] for l in locations])
        locations_itp = np.stack((loc_x_cs(fs), loc_y_cs(fs), loc_z_cs(fs))).transpose()

        if spline_mode == 'cubic':
            rot_x_cs = spi.CubicSpline(key_frames, [r[0] for r in rotations])
            rot_y_cs = spi.CubicSpline(key_frames, [r[1] for r in rotations])
            rot_z_cs = spi.CubicSpline(key_frames, [r[2] for r in rotations])
        elif spline_mode == 'rbf':
            rot_x_cs = spi.Rbf(key_frames, [r[0] for r in rotations])
            rot_y_cs = spi.Rbf(key_frames, [r[1] for r in rotations])
            rot_z_cs = spi.Rbf(key_frames, [r[2] for r in rotations])
        rotations_itp = np.stack((rot_x_cs(fs), rot_y_cs(fs), rot_z_cs(fs))).transpose()

        # rotations = Rotation.from_euler('xyz', rotations, degrees=False)
        # spline = RotationSpline(key_frames, rotations)
        # rotations_itp = spline(fs).as_euler('xyz', degrees=False)

    return locations_itp, rotations_itp


def apply_animation(obj, locations_itp, rotations_itp, name=''):
    if not name: name = str(obj)[-13:-1]
    for k in range(len(locations_itp)):
        loc = locations_itp[k]
        rot = rotations_itp[k]
        _apply_animation(k, obj, loc, rot)


def animation(setup_info, config):
    animation_mode = config['animation_mode']
    num_frames = config['num_frame']
    num_kf = len(setup_info['cam_pose'])

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames

    obj_pose_frames = []
    cam_pose_frames = []

    cam_pose_keyframe = setup_info['cam_pose']
    obj_pose_keyframe = setup_info['dyna_objs_pose']
    obj_list = setup_info['dynamic_objs']
    for obj_idx, _ in enumerate(obj_pose_keyframe):
        obj_pos = [pos for (pos, euler) in obj_pose_keyframe[obj_idx]]
        obj_euler = [euler for (pos, euler) in obj_pose_keyframe[obj_idx]]
        obj = obj_list[obj_idx]
        loc, rot = interpolate_trajectory(num_frames, num_kf, obj_pos, obj_euler, animation_mode=animation_mode, nonlinear_velocity=config['nonlinear_velocity'], spline_mode='rbf')
        apply_animation(obj.blender_obj, loc, rot)
        pose = [[t, r] for t, r in zip(loc, rot)]
        obj_pose_frames.append(pose)
    cam_pos = [pos for (pos, euler) in cam_pose_keyframe]
    cam_euler = [euler for (pos, euler) in cam_pose_keyframe]
    loc, rot = interpolate_trajectory(num_frames, num_kf, cam_pos, cam_euler, animation_mode=animation_mode, nonlinear_velocity=config['nonlinear_velocity'])
    apply_animation(bpy.context.scene.camera, loc, rot)
    cam_pose_frames = [[t, r] for t, r in zip(loc, rot)]

    return obj_pose_frames, cam_pose_frames

