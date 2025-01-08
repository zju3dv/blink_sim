import blenderproc as bproc
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.utility.Utility import UndoAfterExecution, KeyFrame
import argparse
import sys
import os

import bpy
import bmesh
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(__file__+'/../../..'))

import yaml
from mathutils import Euler, Vector
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

from src.blender.animation import animation
from src.hdr2ldr import hdr2ldr
from src.time_utils import catchtime
from src.blender.env import set_engines

from src.blender.utils import get_vertices_xyz, get_vertices_xyz_v2
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

status_DYNAMIC = 0
status_STATIC = -1
status_NO_CAST = -2
status_DEFORMABLE = -3

particle_VISIBLE = 1
particle_OCCLUDED = 0
particle_OUTBOUND = -1
particle_NO_CAST = -2
particle_BEHIND_CAMERA = -3


def init_particle_data(height, width, num_frame):
    # visible 1 / occluded 0 / outbound -1 / no cast -2 / behind camera -3
    particle_status = np.zeros((height, width, num_frame), int)
    particle_track = np.zeros((height, width, num_frame, 3), np.float32)

    return particle_status, particle_track 

def particle_tracking_v2(width, height, occlusion_threshold, frame_index,
                         deform_error_tolerant_factor, particle_config,
                         dyna_status, tracking_points, instance_map0,
                         face_id0, deform_id2name, barycentric0,
                         have_deformable_obj):
    obj_pose_frames = particle_config['obj_pose_frames']
    cam_pose_frames = particle_config['cam_pose_frames']
    depth = particle_config['depth']
    instance = particle_config['instance']
    cam_pose = np.array(cam_pose_frames)
    obj_pose = np.array(obj_pose_frames)

    K = bproc.camera.get_intrinsics_as_K_matrix()

    particle_status, particle_track = init_particle_data(height, width, len(frame_index))

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)
    particle_track[:,:,0,:2] = np.stack([xv, yv], -1)
    particle_track[:,:,0,2] = depth[0]

    particle_status[:,:,0] = particle_VISIBLE

    ideal_one_pass_num = 100
    if have_deformable_obj:
        iter_list = [0] + list(range(1, len(frame_index), ideal_one_pass_num))
    else:
        iter_list = list(range(0, len(frame_index), ideal_one_pass_num))

    for start_index in iter_list:
        if have_deformable_obj and start_index == 0:
            end_index = 1
        else:
            end_index = min(start_index + ideal_one_pass_num, len(frame_index))
        one_pass_num = end_index - start_index

        cam_R = [R.from_euler('xyz', cam_pose[i, 1]).as_matrix() for i in range(start_index, end_index)]
        cam_t = cam_pose[start_index:end_index, 0]
        Tcw = np.zeros((one_pass_num, 4, 4))
        Tcw[:, 0:3, 0:3] = np.einsum('tij->tji', cam_R)
        Tcw[:, 0:3, 3] = -np.einsum('tij,ti->tj', cam_R, cam_t)
        Tcw[:, 3, 3] = 1

        deform_id2mat_world = {}
        for idx in deform_id2name.keys():
            deform_id2mat_world[idx] = np.zeros((one_pass_num, 4, 4))
        for t in range(start_index, end_index):
            with KeyFrame(frame_index[t] + bpy.context.scene.frame_start):
                for idx, name in deform_id2name.items():
                    deform_id2mat_world[idx][t - start_index] = bpy.data.objects[name].matrix_world

        obj_len = len(obj_pose_frames)
        Two = np.zeros((obj_len, one_pass_num, 4, 4))
        Two[:, :] = np.identity(4)
        if obj_len > 0:
            obj_R = [R.from_euler('xyz', obj_pose[i, j, 1]).as_matrix() for i in range(obj_len) for j in range(start_index, end_index)]
            obj_t = obj_pose[:, start_index:end_index, 0]
            Two[:, :, 0:3, 0:3] = np.array(obj_R).reshape((obj_len, one_pass_num, 3, 3))
            Two[:, :, 0:3, 3] = obj_t
            Two[:, :, 3, 3] = 1

        # calculate track
        Two_all = np.zeros((height, width, one_pass_num, 4, 4))
        Two_all[:, :, :] = np.identity(4)
        for v in range(height):
            for u in range(width):
                if dyna_status[v, u] >= status_DYNAMIC:     # if dynamic, transform obj coord to world coord
                    Two_all[v, u] = Two[dyna_status[v, u]]
                elif dyna_status[v, u] == status_DEFORMABLE:
                    Two_all[v, u] = deform_id2mat_world[instance_map0[v, u]]

        Po_aug = np.concatenate((tracking_points, np.ones((height,width,1))), axis=2)
        Po = np.tile(Po_aug[:,:,None,:], (1,1,one_pass_num,1))

        # get each frame obj coord from faceID and barycentric coord
        deform_mask = dyna_status == status_DEFORMABLE
        deform_faceID = face_id0[deform_mask]
        deform_instanceID = instance_map0[deform_mask]
        deform_barycentric = barycentric0[deform_mask]
        deform_len = deform_faceID.shape[0]
        deform_Po_vertices = np.zeros((deform_len, one_pass_num, 3, 3))

        for t in range(start_index, end_index):
            with KeyFrame(frame_index[t] + bpy.context.scene.frame_start):
                eval_obj_table = {}
                for instanceID in np.unique(deform_instanceID):
                    obj_name = deform_id2name[instanceID]
                    obj = bpy.data.objects[obj_name]
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = obj.evaluated_get(depsgraph)
                    eval_obj_table[obj_name] = eval_obj

                for i in range(deform_len):
                    obj_name = deform_id2name[deform_instanceID[i]]
                    face_id = deform_faceID[i]
                    eval_obj = eval_obj_table[obj_name]
                    deform_Po_vertices[i][t-start_index] = get_vertices_xyz_v2(eval_obj, face_id)

        deform_Po = np.einsum('li,ltij->ltj', deform_barycentric, deform_Po_vertices)
        Po[:,:,:,:3][deform_mask] = deform_Po

        Pw = np.einsum('hwtji,hwti->hwtj', Two_all, Po)
        Pc = np.einsum('tji,hwti->hwtj', Tcw, Pw)
        Pc = Pc[:,:,:,0:3]
        depth_cur = -Pc[:,:,:,2]
        Pc = Pc / depth_cur[:,:,:,np.newaxis]
        Pc[:,:,:,1:3] = -Pc[:,:,:,1:3]
        Puv = np.einsum('ji,hwti->hwtj', K, Pc)

        # check deform particle usable through frame 0
        if have_deformable_obj and start_index == 0:
            x = np.linspace(0, width-1, width)
            y = np.linspace(0, height-1, height)
            xv, yv = np.meshgrid(x, y)

            uv_dist = np.linalg.norm(np.stack([xv, yv], -1) - Puv[:,:,0,0:2], axis=-1)
            uv_mask = uv_dist > 1.0

            depth_mask = np.abs(depth_cur[...,0] - depth[0]) > (depth[0] * 0.01)

            error_mask = np.logical_and(np.logical_or(uv_mask, depth_mask), deform_mask)
            dyna_status[error_mask] = status_NO_CAST
            continue

        particle_track[:,:,start_index:end_index,:2] = Puv[:,:,:,0:2]
        particle_track[:,:,start_index:end_index, 2] = depth_cur

        # out of bound check
        out_check = (Puv[:,:,:,0] < 0) | (Puv[:,:,:,0] > width) | (Puv[:,:,:,1] < 0) | (Puv[:,:,:,1] > height)
        particle_status[:,:,start_index:end_index][out_check] = particle_OUTBOUND

        # visible check
        non_occlusion = estimate_occlusion_by_depth_and_segment(
            np.stack([depth[i] for i in range(start_index, end_index)], axis=-1),
            np.stack([instance[i] for i in range(start_index, end_index)], axis=-1),
            Puv[:,:,:,0:2],
            depth_cur * 0.99,
            instance_map0
        )
        non_occlusion = non_occlusion & (~out_check)
        particle_status[:,:,start_index:end_index][non_occlusion] = particle_VISIBLE

        behind_camera = depth_cur <= 0
        particle_status[:,:,start_index:end_index][behind_camera] = particle_BEHIND_CAMERA

    # assign no cast
    no_cast_status = dyna_status == status_NO_CAST
    particle_status[no_cast_status] = particle_NO_CAST

    particle_track = np.transpose(particle_track, (2, 0, 1, 3))
    particle_status = np.transpose(particle_status, (2, 0, 1))

    return particle_status, particle_track

def get_ray(resolution, K):
    height, width = resolution
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    xv, yv = np.meshgrid(x, y)
    # When compared with the rendered optical flow, we find that
    # we do not need to add 0.5 to the pixel coordinates
    # xv = xv + 0.5
    # yv = yv + 0.5
    pixel_direction = np.stack([(xv-cx)/fx, -(yv-cy)/fy, -np.ones_like(xv)], -1) # (height, width, 3)

    return pixel_direction

def get_tracking_points_v3(width, height, setup_info, depth0, instance_map0):
    obj_pose = setup_info['dyna_objs_pose']
    dynamic_objs = setup_info['dynamic_objs']
    cam_pose = setup_info['cam_pose']
    deformable_objs = setup_info['deformable_objs'] # out of date

    Rwc = R.from_euler('xyz', cam_pose[1]).as_matrix()
    twc = cam_pose[0]

    K = bproc.camera.get_intrinsics_as_K_matrix()

    # ray cast through the depth map
    pixel_direction = get_ray((height, width), K)
    Pc = pixel_direction * depth0[..., None]
    Pw = Pc @ Rwc.T + twc

    dyna_status = np.zeros((height, width), int)     # {obj_idx} dyna / -1 static / -2 no cast / -3-{obj_idx} deformable
    tracking_points = np.zeros((height, width, 3), float)   # Po if dyna / Pw if static / Pw if no cast / Pw if deformable

    dyna_id2idx = get_pass_index_to_list_idx(dynamic_objs)
    deform_id2name = get_pass_index_to_obj_name(deformable_objs)

    dyna_idx2Tow = np.zeros((len(obj_pose), 4, 4))
    for idx, pose in enumerate(obj_pose):
        obj_R = R.from_euler('xyz', obj_pose[idx][1]).as_matrix()
        obj_t = obj_pose[idx][0]
        dyna_idx2Tow[idx, 0:3, 0:3] = obj_R.T
        dyna_idx2Tow[idx, 0:3, 3] = -obj_R.T @ obj_t
        dyna_idx2Tow[idx, 3, 3] = 1

    for v in range(height):
        for u in range(width):
            instance_id = instance_map0[v][u]
            hit_location = Pw[v][u]
            tracking_points[v][u] = hit_location
            if instance_id == 0:  # no cast
                dyna_status[v][u] = status_NO_CAST
                continue
            dyna_status[v][u] = status_STATIC
            # check if dynamic
            if instance_id in dyna_id2idx.keys():
                idx = dyna_id2idx[instance_id]
                dyna_status[v][u] = idx
                hit_loc_aug = np.concatenate([hit_location, np.ones(1)])[:, None]
                tracking_points[v][u] = (dyna_idx2Tow[idx] @ hit_loc_aug)[:3, 0]
            # check if deformable
            if instance_id in deform_id2name.keys():
                dyna_status[v][u] = status_DEFORMABLE

    return dyna_status, tracking_points, deform_id2name


def get_pass_index_to_list_idx(objs):
    id2idx = {}
    for idx, obj in enumerate(objs):
        blender_obj = obj.blender_obj
        id2idx[blender_obj.pass_index] = idx
        id2idx.update(get_pass_index_to_list_idx(obj.get_children()))

    return id2idx


def get_pass_index_to_obj_name(objs):
    id2name = {}
    for obj in objs:
        blender_obj = obj.blender_obj
        id2name[blender_obj.pass_index] = blender_obj.name
        id2name.update(get_pass_index_to_obj_name(obj.get_children()))
    return id2name


# https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py#L148
def estimate_occlusion_by_depth_and_segment(depth, segments, uv, depth_reproj, seg_reproj):
    # depth -> [height, width, num_frames]
    # segments -> [height, width, num_frames]
    # uv -> [height, width, num_frames, 2]
    # depth_reproj -> [height, width, num_frames]
    # seg_reproj -> [height, width]

    H, W = depth.shape[:2]

    seg_reproj = seg_reproj[...,None].repeat(depth.shape[-1], axis=-1)
    frame_index = np.arange(depth.shape[-1])[None, None, :].repeat(H, axis=0).repeat(W, axis=1)

    x0 = np.floor(uv[...,0]).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(uv[...,1]).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, a_min=0, a_max=W-1)
    x1 = np.clip(x1, a_min=0, a_max=W-1)
    y0 = np.clip(y0, a_min=0, a_max=H-1)
    y1 = np.clip(y1, a_min=0, a_max=H-1)

    i1 = depth[y0, x0, frame_index]
    i2 = depth[y1, x0, frame_index]
    i3 = depth[y0, x1, frame_index]
    i4 = depth[y1, x1, frame_index]

    depth = np.maximum(np.maximum(np.maximum(i1, i2), i3), i4)

    i1 = segments[y0, x0, frame_index]
    i2 = segments[y1, x0, frame_index]
    i3 = segments[y0, x1, frame_index]
    i4 = segments[y1, x1, frame_index]

    depth_occluded = depth < depth_reproj
    seg_occluded = np.ones_like(depth_occluded, dtype=bool)
    for i in [i1, i2, i3, i4]:
        seg_occluded = np.logical_and(seg_occluded, i != seg_reproj)

    occlusion = np.logical_or(depth_occluded, seg_occluded)
    non_occlusion = np.logical_not(occlusion)

    return non_occlusion
