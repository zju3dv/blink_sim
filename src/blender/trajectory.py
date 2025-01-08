import blenderproc as bproc
import bpy
import random
import tqdm
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Line, Sphere
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
from src.blender.animation import interpolate_trajectory
from src.blender.object_pool import create_pyramid_for_camera


def normaliz_vec(vec):
    vec = vec / np.linalg.norm(vec)
    return vec


def euler_from_look_at(position, target, up):
    forward = np.subtract(target, position)
    forward = np.divide( forward, np.linalg.norm(forward) )

    right = np.cross( forward, up )
    
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array( [0.001, 0, 0] )
        right = np.cross( forward, up + epsilon )
        
    right = np.divide( right, np.linalg.norm(right) )
    
    up = np.cross( right, forward )
    up = np.divide( up, np.linalg.norm(up) )

    T = np.array([[right[0], up[0], -forward[0], position[0]], 
                    [right[1], up[1], -forward[1], position[1]], 
                    [right[2], up[2], -forward[2], position[2]],
                    [0, 0, 0, 1]]) 
    euler = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False)

    return euler




def sphere_seg_inter(start, end, r):
    '''calculate the intersection of a sphere with a certain radius and a line segment'''
    line_vec = end - start
    sphere = Sphere([0, 0, 0], r)
    line = Line(start, line_vec)
    point_a = None
    point_b = None
    try:
        point_a, point_b = sphere.intersect_line(line)
    except:
        pass
    if point_a is not None:
        sa = (point_a - start)/line_vec
        sb = (point_b - start)/line_vec
        sa = sa[~np.isnan(sa)]
        sb = sb[~np.isnan(sb)]
        t = min(np.median(sa),np.median(sb))
    else:
        t = 2.0
    return t

def check_end_clip(objs_radius, obj_pose_list, obj_idx, pos, valid_object):
    '''check whether the motion track coincides with that of other objects'''
    min_t = 1.0
    is_overlap = False
    for check_idx in range(obj_idx):
        if len(obj_pose_list[check_idx]) != 2 or (not valid_object[check_idx]):
            continue
        a0 = obj_pose_list[check_idx][0][0]
        a1 = obj_pose_list[check_idx][1][0]
        b0 = obj_pose_list[obj_idx][0][0]
        b1 = pos
        t = sphere_seg_inter(a0-b0, a1-b1, objs_radius[obj_idx] + objs_radius[check_idx])
        dist = np.linalg.norm(b1-a1)
        if dist < objs_radius[obj_idx] + objs_radius[check_idx]:
            is_overlap = True
        if t>0.0 and t <1.0:
            min_t = min(min_t, t)
            is_overlap = True
    return min_t, is_overlap

def check_start_overlap(objs_radius, obj_pose_list, obj_idx, pos, valid_object):
    '''check whether the starting point coincides with other objects'''
    for check_idx in range(obj_idx):
        if (not valid_object[check_idx]):
            continue
        a0 = obj_pose_list[check_idx][0][0]
        b0 = pos
        dist = np.linalg.norm(b0-a0)
        if dist < objs_radius[obj_idx] + objs_radius[check_idx]:
            return True
    return False

def compute_radius(dynamic_objs, change_origin = True):
    '''calculate the object radius and set the object origin at the object center'''
    objs_radius = []
    for obj in dynamic_objs:
        bb = obj.get_bound_box()
        bb_array = np.array(bb)
        bb_min = np.min(bb_array, axis = 0)
        bb_max = np.max(bb_array, axis = 0)
        if(change_origin):
            obj.set_origin((bb_min+bb_max)/2)
            obj.set_location([0,0,0])
        bb = obj.get_bound_box()
        diag = bb_max - bb_min
        radius = np.linalg.norm(diag)/2
        objs_radius.append(radius)
    return objs_radius


def generate_trajectory(cam_pose_list, dynamic_objs, range_for_dynamic_obj):
    obj_pose_list = [[]] * len(dynamic_objs)
    valid_object = [True]*len(dynamic_objs)
    objs_radius = compute_radius(dynamic_objs)
    for frame_idx, cam_pose in enumerate(cam_pose_list):
        cam_position, look_at = cam_pose[0], cam_pose[1]
        forward_vec = normaliz_vec(np.pad(look_at[:2]-cam_position[:2], (0,1), 'constant'))
        right_vec = np.array([-forward_vec[1], forward_vec[0], 0])
        up_vec = np.array([0, 0, 1])
        look_at = look_at -forward_vec*15 + up_vec*6

        Twb = np.eye(4)
        Twb[:3, 0] = right_vec
        Twb[:3, 1] = forward_vec
        Twb[:3, 2] = up_vec
        Twb[:3, 3] = look_at
        for obj_idx, obj in enumerate(dynamic_objs):
            if not valid_object[obj_idx]:
                continue
            max_t = -1.0
            max_pos = None
            # print('obj_idx:',obj_idx)
            for iter in range(30):
                pos = np.random.uniform(range_for_dynamic_obj[0:3], range_for_dynamic_obj[3:6])
                pos = np.expand_dims(np.pad(pos, (0,1), 'constant', constant_values=1), axis=1)
                pos = (Twb @ pos).squeeze()[:3]
                if len(obj_pose_list[obj_idx]) == 0: # 随机起始点的时候
                    is_overlap = check_start_overlap(objs_radius, obj_pose_list, obj_idx, pos, valid_object)
                    max_pos = pos
                    if not is_overlap:
                        break # 只要和前面的没有重合，就是一个有效的起始点
                else: # 随机终点的时候
                    t, is_overlap = check_end_clip(objs_radius, obj_pose_list, obj_idx, pos, valid_object)
                    max_pos = pos
                    if not is_overlap:
                        break
            if len(obj_pose_list[obj_idx]) == 0 and is_overlap:
                valid_object[obj_idx] = False
            if len(obj_pose_list[obj_idx]) != 0 and is_overlap:
                valid_object[obj_idx] = False
            euler = np.random.uniform([0, 0, 0], [np.pi/5, np.pi/5, np.pi/5]) #np.array([0.0,0.0,0.0])
            if len(obj_pose_list[obj_idx]) == 0: obj_pose_list[obj_idx] = [[max_pos, euler]]
            else: obj_pose_list[obj_idx].append([max_pos, euler])

    valid_dynamic_objs = []
    valid_obj_pose_list = []
    for obj_idx, obj in enumerate(dynamic_objs):
        if valid_object[obj_idx]: 
            valid_dynamic_objs.append(obj)
            valid_obj_pose_list.append(obj_pose_list[obj_idx])

    return valid_dynamic_objs, valid_obj_pose_list

def generate_middle_pt_for_spline(first_frame, second_frame, up=None):
    if len(first_frame) == 3:
        # for camera    
        pos_first, target_first, euler_first = first_frame
        pos_second, target_second, euler_second = second_frame
        vec_len = np.linalg.norm(pos_second - pos_first)
        ratio = 0.1
        radius = vec_len * ratio
        pos_middle = (pos_first + pos_second) / 2 + np.random.uniform([-radius,-radius,-radius], [radius,radius,radius])
        target_middle = (target_first + target_second) / 2
        euler_middle = euler_from_look_at(pos_middle, target_middle, up)
        middle_pose = [pos_middle, target_middle, euler_middle]
    else:
        # for obj
        pos_first, euler_first = first_frame
        pos_second, euler_second = second_frame
        vec_len = np.linalg.norm(pos_second - pos_first)
        ratio = 0.1
        radius = vec_len * ratio
        pos_middle = (pos_first + pos_second) / 2 + np.random.uniform([-radius,-radius,-radius], [radius,radius,radius])
        # rotations = Rotation.from_euler('xyz', [euler_first, euler_second], degrees=False)
        # euler_middle = Slerp([0, 1], rotations)(0.5)
        euler_middle = (euler_first + euler_second) / 2
        middle_pose = [pos_middle, euler_middle]
 
    return middle_pose


def gen_camera_traj(init_pose, prev_up, num_kf, config, mode='linear'):
    world_radius = config['world_radius']
    camera_active_radius = config['camera_active_radius']
    roll_theta_offset = [math.pow(config['roll_theta_offset'][0], 1/config['roll_theta_power']),
                         math.pow(config['roll_theta_offset'][1], 1/config['roll_theta_power'])]
    pos_offset_range, target_offset_range = [[], []], [[], []]
    for i in range(3):
        lower_bound = math.pow(config['pos_offset'][0][i], config['pos_offset_power'])
        upper_bound = math.pow(config['pos_offset'][1][i], config['pos_offset_power'])
        pos_offset_range[0].append(lower_bound)
        pos_offset_range[1].append(upper_bound)
    for i in range(3):
        lower_bound = math.pow(config['target_offset'][0][i], config['target_offset_power'])
        upper_bound = math.pow(config['target_offset'][1][i], config['target_offset_power'])
        target_offset_range[0].append(lower_bound)
        target_offset_range[1].append(upper_bound)

    init_pos, init_target, init_euler = init_pose
    cam_pose_list = [[init_pos, init_target, init_euler]]

    for frame_idx in range(num_kf-1):
        pos_offset = np.random.uniform(pos_offset_range[0], pos_offset_range[1])
        pos_offset[0] = pos_offset[0] if pos_offset[0] < 1 else pos_offset[0]**config['pos_offset_power']
        pos_offset[1] = pos_offset[1] if pos_offset[1] < 1 else pos_offset[1]**config['pos_offset_power']
        sign = 1 if random.random() < 0.5 else -1
        pos_offset[:2] = sign * pos_offset[:2]
        pos = init_pos + pos_offset
        pos[0] = np.clip(pos[0], a_min=-camera_active_radius, a_max=camera_active_radius)
        pos[1] = np.clip(pos[1], a_min=0, a_max=camera_active_radius)
        pos[2] = np.clip(pos[2], a_min=5, a_max=15)

        prev_target = cam_pose_list[-1][1]
        target_offset = np.random.uniform(target_offset_range[0], target_offset_range[1])
        target_offset[0] = target_offset[0] if target_offset[0] < 1 else target_offset[0]**config['target_offset_power']
        target_offset[1] = target_offset[1] if target_offset[1] < 1 else target_offset[1]**config['target_offset_power']
        sign = 1 if random.random() < 0.5 else -1
        target_offset[:2] = sign * target_offset[:2]
        target = prev_target + target_offset
        target[0] = np.clip(target[0], a_min=-world_radius, a_max=world_radius)
        target[1] = np.clip(target[1], a_min=0, a_max=world_radius)
        target[2] = np.clip(target[2], a_min=0, a_max=15)
        target = normaliz_vec(target) * world_radius

        up = np.array([prev_up[0], prev_up[2]])
        up = normaliz_vec(up)
        prev_theta = math.atan2(prev_up[2], prev_up[0])
        theta = np.random.uniform(roll_theta_offset[0], roll_theta_offset[1])
        theta = theta**config['roll_theta_power'] / 180 * np.pi
        sign = 1 if random.random() < 0.5 else -1
        theta = sign * theta
        if prev_theta + theta > np.pi or prev_theta + theta < 0:
            theta = -theta
        rot_mat = np.array([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta), math.cos(theta)]])
        up = rot_mat @ up
        up = np.array([up[0], 0, up[1]])
        up[0] = np.clip(up[0], a_min=-0.95, a_max=0.95)
        up[2] = np.clip(up[2], a_min=0.05, a_max=1)
        up = normaliz_vec(up)
        prev_up = up.copy()

        euler = euler_from_look_at(pos, target, up)

        if mode == 'cubic_spline':
            middle_pose = generate_middle_pt_for_spline(cam_pose_list[-1], [pos, target, euler], up)
            cam_pose_list.append(middle_pose)
        
        cam_pose_list.append([pos, target, euler])
    
    return cam_pose_list


def random_pose_in_cam_frustum(cam_position, cam_look_at, space_range):
    forward_vec = normaliz_vec(np.pad(cam_look_at[:2]-cam_position[:2], (0,1), 'constant'))
    right_vec = np.array([forward_vec[1], -forward_vec[0], 0])
    up_vec = np.array([0, 0, 1])
    look_at = cam_position

    Twb = np.eye(4)
    Twb[:3, 0] = right_vec
    Twb[:3, 1] = forward_vec
    Twb[:3, 2] = up_vec
    Twb[:3, 3] = look_at

    pos = np.random.uniform(space_range[0:3], space_range[3:6])
    pos = np.expand_dims(np.pad(pos, (0,1), 'constant', constant_values=1), axis=1)
    pos = (Twb @ pos).squeeze()[:3]

    return pos


def set_dummy_origin(dynamic_objs):
    obj_origin_list = []
    loc = np.array([0,0,0])
    rot = np.array([0,0,0])
    
    for obj in dynamic_objs:
        # for set origin
        obj.set_location(loc)
        obj.set_rotation_euler(rot)

        bound_box = np.array(obj.get_bound_box())
        x_min, x_max = np.min(bound_box[...,0]), np.max(bound_box[...,0])
        y_min, y_max = np.min(bound_box[...,1]), np.max(bound_box[...,1])
        z_min, z_max = np.min(bound_box[...,2]), np.max(bound_box[...,2])
        origin = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
        origin = np.array([x_max, y_max, z_max])
        obj_origin_list.append(origin.reshape(3, 1))

        obj.set_origin(origin)  # world coord

    return obj_origin_list


def generate_trajectory_v2(cam_pose_list, dynamic_objs, range_for_dynamic_obj,
                           camera_active_radius, static_obj_list, collision_check,
                           rotation_para, tolerate_dist=3, num_attempt=30, 
                           num_seg=10, animation_mode='linear'):
    obj_pose_list = [[]] * len(dynamic_objs)
    obj_interpolated_pose = [[]] * len(dynamic_objs)
    valid_object = [False for _ in range(len(dynamic_objs))]

    # TODO: spline impl
    if animation_mode == 'linear':
        step = 1
    elif animation_mode == 'cubic_spline':
        step = 2

    index = 0
    bvh_cache = {}

    cam_init_pose = cam_pose_list[0]

    # camera_mesh representing the camera
    K_matrix = bproc.camera.get_intrinsics_as_K_matrix()
    resolution = (bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x)
    camera_mesh = create_pyramid_for_camera(K_matrix, resolution, tolerate_dist)
    camera_mesh.set_location(cam_init_pose[0])
    camera_mesh.set_rotation_euler(cam_init_pose[2])

    # set dummy origin for each dynamic obj
    set_dummy_origin(dynamic_objs)
    dummy_mesh = bpy.data.meshes.new("dummy_mesh")
    dummy_obj_for_calculate = bpy.data.objects.new('dummy_obj_for_calculate', dummy_mesh)
    current_scene = bpy.context.scene
    current_scene.collection.objects.link(dummy_obj_for_calculate)

    # init pose for objects
    for obj_idx, obj in enumerate(dynamic_objs):
        for _ in range(num_attempt):
            loc = random_pose_in_cam_frustum(cam_init_pose[0], cam_init_pose[1], range_for_dynamic_obj)
            loc[0] = np.clip(loc[0], a_min=-camera_active_radius, a_max=camera_active_radius)
            loc[1] = np.clip(loc[1], a_min=0, a_max=camera_active_radius)
            rot = R.random().as_euler('xyz')    # initial rotation

            obj.set_location(loc)
            obj.set_rotation_euler(rot)
            if obj.get_name() in bvh_cache:
                del bvh_cache[obj.get_name()]
            no_collision = CollisionUtility.check_intersections(
                obj, bvh_cache, static_obj_list+[camera_mesh], static_obj_list+[camera_mesh])
            if collision_check and not no_collision:
                continue
            obj_pose_list[obj_idx] = [[loc, rot]]
            valid_object[obj_idx] = True
            break

    index += step
    while index < len(cam_pose_list):
        place_object = [False for _ in range(len(dynamic_objs))]
        dyna_list = []
        for obj_idx, obj in enumerate(tqdm.tqdm(dynamic_objs)):
            if not valid_object[obj_idx]:
                continue
            for _ in range(num_attempt):
                pos_prev = obj_pose_list[obj_idx][-1][0]
                euler_prev = obj_pose_list[obj_idx][-1][1]

                cam_pose = cam_pose_list[index]
                pos = random_pose_in_cam_frustum(cam_pose[0], cam_pose[1], range_for_dynamic_obj)
                pos[0] = np.clip(pos[0], a_min=-camera_active_radius, a_max=camera_active_radius)
                pos[1] = np.clip(pos[1], a_min=0, a_max=camera_active_radius)

                rotation_axis = R.random().as_rotvec()
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_para_idx = np.random.choice(np.arange(rotation_para[:,:2].shape[0]), p=rotation_para[:,2])
                rotation_angle = np.random.uniform(rotation_para[rotation_para_idx,0], rotation_para[rotation_para_idx,1])
                rotation_iter_time = int(rotation_angle // np.pi + 1)
                rotation_angle /= rotation_iter_time    # split to [0, pi]
                if np.random.randint(2):
                    rotation_angle = -rotation_angle
                delta_euler = R.from_rotvec(rotation_axis * rotation_angle).as_euler('xyz')

                euler = euler_prev
                for iter_idx in range(rotation_iter_time):
                    # add euler
                    dummy_obj_for_calculate.rotation_euler = euler
                    dummy_obj_for_calculate.delta_rotation_euler = delta_euler
                    bpy.ops.object.select_all(action='DESELECT')
                    dummy_obj_for_calculate.select_set(True)
                    bpy.ops.object.transforms_to_deltas(mode='ROT')
                    euler = np.array(dummy_obj_for_calculate.delta_rotation_euler)

                pos_list = [pos_prev, pos]
                euler_list = [euler_prev, euler]
                if animation_mode == 'cubic_spline':
                    pos_middle, euler_middle = generate_middle_pt_for_spline([pos_list[0], euler_list[0]], [pos_list[1], euler_list[1]])
                    pos_list = [pos_list[0], pos_middle, pos_list[1]]
                    euler_list = [euler_list[0], euler_middle, euler_list[1]]
                loc, rot = interpolate_trajectory(num_seg, 1 + step, pos_list, euler_list, animation_mode=animation_mode, spline_mode='rbf')

                # interpolate trajectory for cameras
                if animation_mode == 'linear':
                    camera_pos_list = [cam_pose_list[index-1][0], cam_pose_list[index][0]]
                    camera_euler_list = [cam_pose_list[index-1][2], cam_pose_list[index][2]]
                elif animation_mode == 'cubic_spline':
                    camera_pos_list = [cam_pose_list[index-2][0], cam_pose_list[index-1][0], cam_pose_list[index][0]]
                    camera_euler_list = [cam_pose_list[index-2][2], cam_pose_list[index-1][2], cam_pose_list[index][2]]
                camera_loc, camera_rot = interpolate_trajectory(num_seg, 1 + step, camera_pos_list, camera_euler_list, animation_mode=animation_mode)

                seg_success = True
                for i_seg in range(num_seg):
                    for _index in range(len(place_object)):
                        if not place_object[_index]:
                            continue
                        if dynamic_objs[_index].get_name() in bvh_cache:
                            del bvh_cache[dynamic_objs[_index].get_name()]
                        dynamic_objs[_index].set_location(obj_interpolated_pose[_index][i_seg][0])
                        dynamic_objs[_index].set_rotation_euler(obj_interpolated_pose[_index][i_seg][1])

                    obj.set_location(loc[i_seg])
                    obj.set_rotation_euler(rot[i_seg])
                    if obj.get_name() in bvh_cache:
                        del bvh_cache[obj.get_name()]

                    camera_mesh.set_location(camera_loc[i_seg])
                    camera_mesh.set_rotation_euler(camera_rot[i_seg])
                    if camera_mesh.get_name() in bvh_cache:
                        del bvh_cache[camera_mesh.get_name()]

                    no_collision = CollisionUtility.check_intersections(
                        obj, bvh_cache, static_obj_list+dyna_list+[camera_mesh], static_obj_list+dyna_list+[camera_mesh])
                    if collision_check and not no_collision:
                        seg_success = False
                        break
 
                if seg_success:
                    place_object[obj_idx] = True
                    dyna_list.append(dynamic_objs[obj_idx])
                    obj_pose_list[obj_idx] += [[p, r] for p, r in zip(pos_list[1:], euler_list[1:])]
                    obj_interpolated_pose[obj_idx] = [[p, r] for p, r in zip(loc, rot)]
                    break

            if not place_object[obj_idx]:
                valid_object[obj_idx] = False

        index += step

    camera_mesh.delete()
    bpy.data.objects.remove(dummy_obj_for_calculate, do_unlink=True)
    bpy.data.meshes.remove(dummy_mesh)

    valid_dynamic_objs = []
    valid_obj_pose_list = []
    for obj_idx, obj in enumerate(dynamic_objs):
        if valid_object[obj_idx]: 
            valid_dynamic_objs.append(obj)
            valid_obj_pose_list.append(obj_pose_list[obj_idx])
        else:
            obj.delete()

    return valid_dynamic_objs, valid_obj_pose_list

