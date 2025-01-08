import bpy
import numpy as np
import tqdm
from src.blender.particle import get_tracking_points_v3, particle_tracking_v2
from blenderproc.python.utility.Utility import KeyFrame

def compute_video_corr(ref_range, target_range, depth, instance_segmaps,
                       obj_pose_frames, cam_pose_frames, width, height, num_frame,
                       rigid_tracked_obj_list, deformable_obj_list,
                       occlusion_threshold, deform_error_tolerant_factor,
                       face_id=None, barycentric=None):
    print('compute video corr')
    particle_track, particle_status = [], []
    # collect the target frame for the same ref frame
    pairs = {}
    for i, (ref_frame, target_frame) in enumerate(zip(ref_range, target_range)):
        if ref_frame not in pairs:
            pairs[ref_frame] = {}
            pairs[ref_frame]['index'] = [i]
            pairs[ref_frame]['target'] = [target_frame]
        else:
            pairs[ref_frame]['index'].append(i)
            pairs[ref_frame]['target'].append(target_frame)
    
    indices = []
    for ref_frame in tqdm.tqdm(pairs.keys()):
        target_frames = pairs[ref_frame]['target']
        index_list = pairs[ref_frame]['index']
        depth_ref = depth[ref_frame]
        instance_map_ref = instance_segmaps[ref_frame]
        face_id_ref = face_id[ref_frame].astype(np.int32) if face_id is not None else np.zeros((height, width), dtype=np.int32)
        barycentric_ref = barycentric[ref_frame]  if barycentric is not None else np.zeros((height, width, 3), dtype=np.float32)
        barycentric_ref[:,:,2] = 1 - barycentric_ref[:,:,0] - barycentric_ref[:,:,1]
        tracking_config = {
            'dyna_objs_pose': [obj_pose_frames[i][ref_frame] for i in range(len(obj_pose_frames))],
            'dynamic_objs': rigid_tracked_obj_list,
            'deformable_objs': deformable_obj_list,
            'cam_pose': cam_pose_frames[ref_frame],
        }
        with KeyFrame(bpy.context.scene.frame_start + ref_frame):
            dyna_status, tracking_points, deform_id2name = get_tracking_points_v3(width, height, tracking_config,
                                                                                  depth_ref, instance_map_ref)
        
        have_deformable_obj = len(deformable_obj_list) > 0
        if have_deformable_obj:
            frame_index = [ref_frame] + target_frames
        else:
            frame_index = target_frames
        particle_config = {
            'obj_pose_frames': [[obj_pose_frames[obj_idx][i] for i in frame_index] for obj_idx in range(len(obj_pose_frames))],
            'cam_pose_frames': [cam_pose_frames[i] for i in frame_index],
            'depth': [depth[i] for i in frame_index],
            'instance': [instance_segmaps[i] for i in frame_index],
        }
        # part_of_status, part_of_track = particle_tracking(config, particle_config, dyna_status, tracking_points)
        part_of_status, part_of_track = particle_tracking_v2(width, height, occlusion_threshold, frame_index,
                                                             deform_error_tolerant_factor, particle_config,
                                                             dyna_status, tracking_points,
                                                             instance_map_ref, face_id_ref,
                                                             deform_id2name, barycentric_ref,
                                                             have_deformable_obj=have_deformable_obj)
        if have_deformable_obj:
            part_of_status = part_of_status[1:]
            part_of_track = part_of_track[1:]

        part_of_status = np.split(part_of_status, len(index_list), axis=0)
        part_of_track = np.split(part_of_track, len(index_list), axis=0)
        particle_status += part_of_status
        particle_track += part_of_track

        indices += index_list

    particle_status = [particle_status[indices.index(i)][0] for i in range(len(indices))]
    particle_track = [particle_track[indices.index(i)][0] for i in range(len(indices))]

    return particle_status, particle_track


def particle_to_flow(track_list):
    # track_list: list of HxWx3 numpy
    flow_list = []
    H, W, _ = track_list[0].shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack((x, y), axis=-1)
    for track in track_list:
        flow = track[..., :2] - coords
        flow_list.append(flow)
    return flow_list


def depth2pc(coords, depth, fx, fy, cx, cy):
    xx, yy = coords[..., 0], coords[..., 1]
    x = (xx - cx) * depth / fx
    y = (yy - cy) * depth / fy

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def particle_to_scene_flow(track_list, depth_list, intrinsic, ref_frame_list):
    H, W, _ = track_list[0].shape
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack((x, y), axis=-1)
    flow3d_list = []
    for i, track in enumerate(track_list):
        ref_index = ref_frame_list[i]
        depth1 = depth_list[ref_index]
        depth2 = track[..., 2]
        track_pos = track[..., :2]
        pt1 = depth2pc(coords, depth1, fx, fy, cx, cy)
        pt2 = depth2pc(track_pos, depth2, fx, fy, cx, cy)
        flow_3d = pt2 - pt1
        flow3d_list.append(flow_3d)

    return flow3d_list

def compute_flexible_particle(stop_ratio, start_ratio, depth, instance_segmaps,
                       obj_pose_frames, cam_pose_frames, width, height, num_frame,
                       rigid_tracked_obj_list, deformable_obj_list,
                       occlusion_threshold, deform_error_tolerant_factor,
                       face_id=None, barycentric=None, consecutive_frame=5):
    print('compute flexible particle')
    assert stop_ratio <= start_ratio
    particle_track, particle_status = [], []
    tracks = []
    ref_queue = [0]
    while len(ref_queue) > 0:
        ref_frame = ref_queue.pop(0)
        depth_ref = depth[ref_frame]
        instance_map_ref = instance_segmaps[ref_frame]
        face_id_ref = face_id[ref_frame].astype(np.int32) if face_id is not None else np.zeros((height, width), dtype=np.int32)
        barycentric_ref = barycentric[ref_frame]  if barycentric is not None else np.zeros((height, width, 3), dtype=np.float32)
        barycentric_ref[:,:,2] = 1 - barycentric_ref[:,:,0] - barycentric_ref[:,:,1]
        tracking_config = {
            'dyna_objs_pose': [obj_pose_frames[i][ref_frame] for i in range(len(obj_pose_frames))],
            'dynamic_objs': rigid_tracked_obj_list,
            'deformable_objs': deformable_obj_list,
            'cam_pose': cam_pose_frames[ref_frame],
        }
        with KeyFrame(bpy.context.scene.frame_start + ref_frame):
            dyna_status, tracking_points, deform_id2name = get_tracking_points_v3(width, height, tracking_config,
                                                                                  depth_ref, instance_map_ref)

        have_deformable_obj = len(deformable_obj_list) > 0

        prev_index = ref_frame
        stop_index, restart_index = -1, -1
        while True:
            target_frames = list(range(prev_index, min(prev_index + consecutive_frame, num_frame)))

            if len(target_frames) < 1:
                tracks.append([ref_frame, min(prev_index, num_frame)])
                break

            if have_deformable_obj:
                frame_index = [ref_frame] + target_frames
            else:
                frame_index = target_frames
    
            particle_config = {
                'obj_pose_frames': [[obj_pose_frames[obj_idx][i] for i in frame_index] for obj_idx in range(len(obj_pose_frames))],
                'cam_pose_frames': [cam_pose_frames[i] for i in frame_index],
                'depth': [depth[i] for i in frame_index],
                'instance': [instance_segmaps[i] for i in frame_index],
            }
            part_of_status, part_of_track = particle_tracking_v2(width, height, occlusion_threshold, frame_index,
                                                                deform_error_tolerant_factor, particle_config,
                                                                dyna_status, tracking_points,
                                                                instance_map_ref, face_id_ref,
                                                                deform_id2name, barycentric_ref,
                                                                have_deformable_obj=have_deformable_obj)
            if have_deformable_obj:
                part_of_status = part_of_status[1:]
                part_of_track = part_of_track[1:]

            part_of_status = [item[0] for item in np.split(part_of_status, len(target_frames), axis=0)]
            part_of_track = [item[0] for item in np.split(part_of_track, len(target_frames), axis=0)]

            for i in range(len(part_of_status)):
                valid_ratio = np.sum(part_of_status[i] >= 0) / part_of_status[i].size
                if restart_index == -1 and valid_ratio < start_ratio:
                    restart_index = i
                    ref_queue.append(prev_index + restart_index)
                    restart_index += prev_index
                if stop_index == -1 and valid_ratio < stop_ratio:
                    stop_index = i + 1
                    part_of_status = part_of_status[:stop_index]
                    part_of_track = part_of_track[:stop_index]
                    tracks.append([ref_frame, prev_index + stop_index])
                    stop_index += prev_index
                    break

            particle_status += part_of_status
            particle_track += part_of_track

            prev_index += consecutive_frame

            if stop_index != -1:
                break

    print(f'flexible particle have {len(tracks)} tracks in total')

    return particle_status, particle_track, tracks

