import blenderproc as bproc
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from skspatial.objects import Line, Sphere
import argparse, bpy, math, pickle, cv2, os, sys, shutil
import numpy as np
from blenderproc.python.material.MaterialLoaderUtility import convert_to_materials
sys.path.insert(0, os.path.abspath(__file__+'/../../..'))
from src.blender.movement import animation
from src.hdr2ldr import hdr2ldr, tone_mapping
from src.utils import safe_sample
import yaml
import json
import random
from pathlib import Path
from blenderproc.python.utility.Utility import Utility
from scipy.spatial.transform import Rotation as R
import glob

config = None

def sample_pose(obj: bproc.types.MeshObject):
    global config
    world_length = config['world_length']
    world_width = config['world_width']

    # Sample the spheres location above the surface
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))
    obj.set_location([0,0,0])
    bbox = obj.get_bound_box()
    min_x, max_x = min(bbox[:,0]), max(bbox[:,0])
    min_y, max_y = min(bbox[:,1]), max(bbox[:,1])
    min_z, max_z = min(bbox[:,2]), max(bbox[:,2])
    min_z = min_z if min_z < 0 else 0
    min_x, max_x = -world_length - min_x, world_length - max_x
    min_y, max_y = -world_width - min_y, world_width - max_y
    min_height, max_height = 1+abs(min_z), 4+abs(min_z)
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    z = random.uniform(min_height, max_height)
    obj.set_location([x,y,z])




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

def check_obj_hit(obj, origin, direction, max_distance):
    world2local = np.linalg.inv(obj.get_local2world_mat())
    origin_in_obj = world2local@origin.T
    dir_in_obj = world2local@direction.T
    origin_in_obj = origin_in_obj[:3] * obj.get_scale(0)
    dir_in_obj = dir_in_obj[:3] * obj.get_scale(0)
    hit, location, _, _ = obj.blender_obj.ray_cast(origin_in_obj, dir_in_obj, distance=max_distance)
    return hit, location

def filter_range(ori,objs_list, r):
    '''filter static objects within a certain range of the camera'''
    r_list = compute_radius(objs_list, change_origin = False)
    filter_list = []
    for index, obj in enumerate(objs_list):
        pos = obj.get_location()
        dist = np.linalg.norm(pos-ori)
        # print(dist)
        if dist-r_list[index] > r:
            filter_list.append(obj)
        else:
            obj.set_location([1e3,1e3,1e3])
    return filter_list

def test_obj_hist():
    obj = bproc.object.create_primitive(shape='CUBE')
    obj.set_location([0,0,100])
    obj.set_scale([2,3,4])
    obj.set_rotation_euler(np.array([-0.4636476, 0.111341, -0.4636476]))
    origin = np.array([0,5,100,1])
    direction = np.array([0,-1,0,0])
    for max_distance in range(0, 7):
        hit, location = check_obj_hit(obj, origin, direction, max_distance)
        print(max_distance)
        print(hit)
        print(location)
    exit(0)

def setup_placement(init_pose):
    global config
    world_length = config['world_length']
    world_width = config['world_width']
    canopy_distance = config['canopy_distance']
    canopy_height = config['canopy_height']

    # test_obj_hist()

    ground_plane = bproc.object.create_primitive(shape='PLANE')
    ground_plane.set_scale([world_length+canopy_distance,world_width+canopy_distance,1])
    ground_plane.set_location([0,0,0])
    # remove_shadow(ground_plane)
    objs_list = []
    num_static_obj = random.randint(*config['num_static_obj'])
    for i in range(num_static_obj):
        if random.random() < 0.5:
            obj = bproc.object.create_primitive(shape='CYLINDER')
        else:
            obj = bproc.object.create_primitive(shape='CUBE')
        if random.random() < 0.5:
            random_length = random.uniform(1.0, 4.0)
            random_height = random.uniform(1.0, 8.0)
        else:
            random_length = random.uniform(1.0, 8.0)
            random_height = random.uniform(1.0, 4.0)
        obj.set_scale([random_length, random_length, random_height])
        # remove_shadow(obj)
        objs_list.append(obj)

    # for obj in objs_list:
    #     sample_pose(obj)
    bproc.object.sample_poses_on_surface(
        objs_list,
        ground_plane,
        sample_pose,
        min_distance=0.1,
        max_distance=400,
        up_direction=[0,0,1]
    )

    for obj in objs_list:
        obj.enable_rigidbody(True)
    ground_plane.enable_rigidbody(False)

    # Run the physics simulation
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=2,
        max_simulation_time=8,
        check_object_interval=5
    )

    objs_list = filter_range(init_pose[0], objs_list, 20)

    # stitching ground plane
    bpy.data.objects.remove(ground_plane.blender_obj, do_unlink=True)
    plane_list = []
    # part_num = random.randint(1, 10)
    part_num = int(random.uniform(1, 3) ** 2)
    for i in range(part_num):
        for j in range(part_num):
            plane = bproc.object.create_primitive(shape='PLANE')
            length = math.ceil((world_length+canopy_distance)/part_num)
            width = math.ceil((world_width+canopy_distance)/part_num)
            plane.set_scale([length, width, 1])
            y = (2*i+1-part_num) * length
            x = (2*j+1-part_num) * width
            plane.set_location([y,x,0])
            plane_list.append(plane)

    canopy_list = []
    _scale_list = [
        [math.ceil((world_length+canopy_distance)/part_num),int(canopy_height/2),1],
        [math.ceil((world_length+canopy_distance)/part_num),int(canopy_height/2),1],
        [int(canopy_height/2),math.ceil((world_width+canopy_distance)/part_num),1],
        [int(canopy_height/2),math.ceil((world_width+canopy_distance)/part_num),1],
    ]
    _loc_list = [
        [0,-world_width-canopy_distance,int(canopy_height/2)],
        [0,world_width+canopy_distance,int(canopy_height/2)],
        [-world_length-canopy_distance,0,int(canopy_height/2)],
        [world_length+canopy_distance,0,int(canopy_height/2)],
    ]
    _rot_list = [
        [math.pi/2,0,0],
        [-math.pi/2,0,0],
        [0,-math.pi/2,0],
        [0,math.pi/2,0],
    ]
    for i in range(4):
        for j in range(part_num):
            length = math.ceil((world_length+canopy_distance)/part_num)
            y = (2*j+1-part_num) * length
            loc = _loc_list[i]
            loc = [y if math.fabs(item) < 1e-6 else item for item in loc]

            canopy = bproc.object.create_primitive(shape='PLANE')
            canopy.set_scale(_scale_list[i])
            # canopy.set_location(_loc_list[i])
            canopy.set_location(loc)
            canopy.set_rotation_euler(_rot_list[i])
            # remove_shadow(canopy)
            canopy_list.append(canopy)

    # return ground_plane, objs_list, canopy_list
    return plane_list, objs_list, canopy_list

def remove_shadow(obj):
    if obj.has_materials():
        for i in range(len(obj.blender_obj.data.materials)):
            obj.blender_obj.data.materials[i].shadow_method = 'NONE'
    else:
        obj.new_material("material_0")
        obj.blender_obj.data.materials[0].shadow_method = 'NONE'

def setup_material(ground, static_objs, canopys, mode):
    global config
    image_dir = Path(config['image_dir'])
    texture_split_file = config['texture_split_file']
    
    with open(texture_split_file, 'r') as fr:
        rel_path = json.load(fr)[mode]
        images = [image_dir.joinpath(rp) for rp in rel_path]
    skip_num = 1 + len(canopys)
    # objs = [ground] + canopys + static_objs
    objs = ground + canopys + static_objs
    for index, obj in enumerate(objs):
        if not obj.has_materials():
            obj.new_material("material_0")
        material_0 = obj.get_materials()[0]
        # add image texture as base color
        image = bpy.data.images.load(filepath=str(random.choice(images)))
        material_0.set_principled_shader_value("Base Color", image)
        # insert hsv node, to post-processing texture in hsv color space
        # do not process ground plane
        if index < skip_num: continue
        hsv_node = material_0.new_node(node_type="ShaderNodeHueSaturation")
        random_mode = random.random()
        min_hsv_range = config['min_hsv_range']
        max_hsv_range = config['max_hsv_range']
        hsv_dark_prob = config['hsv_dark_prob']
        if random_mode < hsv_dark_prob: # dark
            random_hsv_value = random.uniform(min_hsv_range[0], min_hsv_range[1])
        else: # bright
            random_hsv_value = random.uniform(max_hsv_range[0], max_hsv_range[1])
        hsv_node.inputs["Value"].default_value = random_hsv_value
        src_node = material_0.get_the_one_node_with_type("TexImage")
        dst_node = material_0.get_the_one_node_with_type("BsdfPrincipled")
        Utility.insert_node_instead_existing_link(
            material_0.links,
            src_node.outputs["Color"],
            hsv_node.inputs["Color"],
            hsv_node.outputs["Color"],
            dst_node.inputs["Base Color"]
        )

def setup_lighting(init_pose):
    # tri-light setup
    up = np.array([0, 0, 1])

    pos0 = np.array([42, -50, 32])
    target0 = np.array([10, -30, 0])
    light0 = bproc.types.Light()
    light0.set_type("AREA")
    light0.set_location(pos0)
    light0.set_rotation_euler(euler_from_look_at(pos0, target0, up))
    light0.set_energy(10e4)

    pos1 = np.array([-64, -13, 40])
    target1 = np.array([-30, 5, 0])
    light1 = bproc.types.Light()
    light1.set_type("AREA")
    light1.set_location(pos1)
    light1.set_rotation_euler(euler_from_look_at(pos1, target1, up))
    light1.set_energy(5e4)

    pos2 = np.array([-20, 58, 40])
    target2 = np.array([-10, 30, 0])
    light2 = bproc.types.Light()
    light2.set_type("AREA")
    light2.set_location(pos2)
    light2.set_rotation_euler(euler_from_look_at(pos2, target2, up))
    light2.set_energy(1e4)

    pos, target, euler = init_pose
    inner_rad = random.uniform(1, 3)
    outer_rad = random.uniform(3, 15)
    _alpha1 = random.uniform(0, 2*math.pi)
    target_offset = np.array([math.cos(_alpha1)*inner_rad, math.sin(_alpha1)*inner_rad, random.uniform(0,3)])
    _alpha2 = random.uniform(0, 2*math.pi)
    pos_offset = np.array([math.cos(_alpha2)*outer_rad, math.sin(_alpha2)*outer_rad, random.uniform(20,40)])
    light3 = bproc.types.Light()
    light3.set_type("AREA")
    light3.set_location(pos+pos_offset)
    light3.set_rotation_euler(euler_from_look_at(pos+pos_offset, pos+target_offset, up))
    light3.set_energy(3e4)


def normalized(a):
    length = np.linalg.norm(a)
    return a / length

def spline_from_3pose(init_pose, up):
    pos, target, euler = init_pose
    cam_pose_list = [[pos, target, euler]]

    # last pose
    _alpha = random.uniform(0, 2*math.pi)
    _theta = random.uniform(0, math.pi/5*2)
    _z = math.cos(_theta)
    _xy_len = math.sin(_theta)
    _x, _y = math.cos(_alpha)*_xy_len, math.sin(_alpha)*_xy_len
    up_new = np.array([_x, _y, _z])
    pos_offset = np.random.uniform([2.0,2.0,2.0], [6.0,6.0,6.0])
    target_offset = np.random.uniform([1.0,1.0,1.0], [3.0,3.0,3.0])

    _pos = pos + pos_offset/2 + np.random.uniform([-1.0,-1.0,-1.0], [1.0,1.0,1.0])
    _target = target + target_offset/2 + np.random.uniform([-0.5,-0.5,-0.5], [0.5,0.5,0.5])
    _up = normalized((up + up_new) / 2)
    euler = euler_from_look_at(_pos, _target, _up)
    cam_pose_list.append([_pos, _target, euler])

    _pos = pos + pos_offset
    _target = target + target_offset
    _up = up_new
    euler = euler_from_look_at(_pos, _target, _up)
    cam_pose_list.append([_pos, _target, euler])

    return cam_pose_list

def linear_pose(init_pose, up, num_kf):
    pos, target, euler = init_pose
    cam_pose_list = [[pos, target, euler]]

    min_height = 12-3
    max_height = 12+5
    for frame_idx in range(num_kf-1):
        # pos_offset = np.random.uniform([-6.0,-6.0,-3.0], [6.0,6.0,3.0])
        # target_offset = np.random.uniform([-2.0,-2.0,0.0], [2.0,2.0,0.0])
        pos_offset = np.random.uniform([0.0,0.0,-3.0], [5.0,5.0,3.0])
        target_offset = np.random.uniform([-2.0,-2.0,0.0], [2.0,2.0,0.0])
        pos_offset[0] = pos_offset[0] if pos_offset[0] < 1 else pos_offset[0]**2
        pos_offset[1] = pos_offset[1] if pos_offset[1] < 1 else pos_offset[1]**2
        sign = 1 if random.random() < 0.5 else -1
        pos_offset[:2] = sign * pos_offset[:2]
        pos = pos + pos_offset
        target = target + target_offset
        pos[2] = np.clip(pos[2], a_min=min_height, a_max=max_height)
        euler = euler_from_look_at(pos, target, up)
        cam_pose_list.append([pos, target, euler])
    
    return cam_pose_list

def setup_camera_extrinsic():
    global config
    num_kf = config['num_keyframes']

    theta = random.uniform(0, math.pi*2)
    radius = 49
    height = 12
    look_at_radius = radius - 35

    # init pose
    pos = np.array([radius*math.sin(theta), radius*math.cos(theta), height])
    target = np.array([look_at_radius*math.sin(theta), look_at_radius*math.cos(theta), 0])
    up = np.array([0, 0, 1])
    euler = euler_from_look_at(pos, target, up)
    init_pose = [pos, target, euler]

    if num_kf == 3 and config['animation_mode'] == 'cubinc_spline':
        cam_pose_list = spline_from_3pose(init_pose, up)
    else:
        cam_pose_list = linear_pose(init_pose, up, num_kf)

    return cam_pose_list

def load_taxonomy_list(path):
    name = f'{path}/taxonomy.json'
    l = []
    with open(name, 'r') as fr:
        d = json.load(fr)
        for i in range(len(d)):
            if d[i]['numInstances'] < 500:
                l.append(d[i]['synsetId'])
    return l

def correct_materials(obj):
    """ If the used material contains an alpha texture, the alpha texture has to be flipped to be correct

    :param obj: object where the material maybe wrong
    """
    for material in obj.get_materials():
        if material is None:
            continue
        texture_nodes = material.get_nodes_with_type("ShaderNodeTexImage")
        if texture_nodes and len(texture_nodes) > 1:
            principled_bsdf = material.get_the_one_node_with_type("BsdfPrincipled")
            # find the image texture node which is connect to alpha
            node_connected_to_the_alpha = None
            for node_links in principled_bsdf.inputs["Alpha"].links:
                if "ShaderNodeTexImage" in node_links.from_node.bl_idname:
                    node_connected_to_the_alpha = node_links.from_node
            # if a node was found which is connected to the alpha node, add an invert between the two
            if node_connected_to_the_alpha is not None:
                invert_node = material.new_node("ShaderNodeInvert")
                invert_node.inputs["Fac"].default_value = 1.0
                material.insert_node_instead_existing_link(node_connected_to_the_alpha.outputs["Color"],
                                                            invert_node.inputs["Color"],
                                                            invert_node.outputs["Color"],
                                                            principled_bsdf.inputs["Alpha"])

def load_shapenet_obj(filepath, move_object_origin=True):
    loaded_objects = bproc.loader.load_obj(filepath)

    # In shapenet every .obj file only contains one object, make sure that is the case
    if len(loaded_objects) != 1:
        raise Exception("The ShapeNetLoader expects every .obj file to contain exactly one object")
    obj = loaded_objects[0]

    correct_materials(obj)

    # removes the x axis rotation found in all ShapeNet objects, this is caused by importing .obj files
    # the object has the same pose as before, just that the rotation_euler is now [0, 0, 0]
    obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)

    # check if the move_to_world_origin flag is set
    if move_object_origin:
        # move the origin of the object to the world origin and on top of the X-Y plane
        # makes it easier to place them later on, this does not change the `.location`
        obj.move_origin_to_bottom_mean_point()
    bpy.ops.object.select_all(action='DESELECT')

    return obj

def filter_obj_by_volume(obj_list, thres=0.80, keep_num=5):
    filter_list = []
    for obj in obj_list:
        bb = obj.get_bound_box()
        min_point, max_point = bb[0], None
        max_dist = -1
        for point in bb:
            dist = np.linalg.norm(point - min_point)
            if dist > max_dist:
                max_point = point
                max_dist = dist
        diag = max_point - min_point
        if diag.max() < thres:
            filter_list.append(obj)
    return filter_list[:keep_num]

def normaliz_vec(vec):
    vec = vec / np.linalg.norm(vec)
    return vec

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

def setup_dynamic_objs(cam_pose_list, mode):
    global config
    activte_range_for_dynamic_obj = config['activte_range_for_dynamic_obj']
    dynamic_obj_split_file = config['dynamic_obj_split_file']
    shape_dir = Path(config['shape_dir'])
    # shape_dir = config['shape_dir']
    
    with open(dynamic_obj_split_file, 'r') as fr:
        rel_path = json.load(fr)[mode]
        all_shapes_p = [shape_dir.joinpath(rp) for rp in rel_path]

    shape_num = random.randrange(config['shape_num'][0], config['shape_num'][1])
    shapes_p = safe_sample(all_shapes_p, shape_num * 3)
    dynamic_objs = [load_shapenet_obj(str(shape_p)) for shape_p in shapes_p]
    dynamic_objs = filter_obj_by_volume(dynamic_objs, thres=0.8, keep_num=shape_num * 3)

    obj_pose_list = [[]] * len(dynamic_objs)
    valid_object = [True]*len(dynamic_objs)
    for obj in dynamic_objs:
        obj.set_scale([random.uniform(5.0, 8.0)]*3)
        obj.enable_rigidbody(True)
        # remove_shadow(obj)
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
                pos = np.random.uniform([-7, -5, -5], activte_range_for_dynamic_obj)
                #pos = np.random.uniform(-activte_range_for_dynamic_obj, activte_range_for_dynamic_obj)
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

def setup_camera_intrinsic():
    global config
    width, height = config['image_width'], config['image_height']
    bproc.camera.set_resolution(width, height)

def setup_envmap():
    global config
    hdr_dir = config['hdr_dir']
    hdr_list = os.listdir(hdr_dir)
    hdr_name = random.choice(hdr_list)
    path = f'{hdr_dir}/{hdr_name}'
    bproc.world.set_world_background_hdr_img(path)

def setup_env(mode):
    cam_pose_list = setup_camera_extrinsic()
    init_pose = cam_pose_list[0]
    ground, static_objs, canopys = setup_placement(init_pose)
    setup_material(ground, static_objs, canopys, mode)
    # setup_lighting(cam_pose_list[0])
    setup_envmap()
    setup_camera_intrinsic()
    pos_lookat_list = [[pos, lookat] for (pos, lookat, euler) in cam_pose_list]
    pos_euler_list = [[pos, euler] for (pos, lookat, euler) in cam_pose_list]
    dynamic_objs, obj_pose_list = setup_dynamic_objs(pos_lookat_list, mode)

    setup_info = {
        'cam_pose': pos_euler_list,
        'dynamic_objs': dynamic_objs,
        'dyna_objs_pose': obj_pose_list,
    }

    return setup_info

def main():
    global config 
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file')
    parser.add_argument('-output_dir')
    parser.add_argument('-mode')
    args = parser.parse_args()

    output_dir = args.output_dir
    config_file = args.config_file
    mode = args.mode

    os.system(f'mkdir -p {output_dir}/hdf5/slow')
    os.system(f'mkdir -p {output_dir}/hdf5/fast')
    os.system(f'mkdir -p {output_dir}/tmp')

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config['activte_range_for_dynamic_obj'] = np.array(config['activte_range_for_dynamic_obj'] )
        except yaml.YAMLError as exc:
            print(exc)

    bproc.init()

    setup_info = setup_env(mode)

    ########### first pass, render motion blur ########### 
    animation(output_dir, setup_info, {
        'animation_mode': config['animation_mode'],
        'num_frame': int(config['rgb_image_fps'] * config['duration']),
        'num_keyframes': config['num_keyframes'],
    })

    # exr format which allows linear colorspace
    bproc.renderer.set_output_format("OPEN_EXR", 16)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 64
    bproc.renderer.set_cpu_threads(1)

    # TODO: Currently we only use slow fps RGB image for ref video, so close motion blur simulation
    bpy.context.scene.render.use_motion_blur = False
    # bproc.renderer.enable_motion_blur(motion_blur_length=0.0)
    blur_data = bproc.renderer.render(f'{output_dir}/tmp')
    # TODO: tmp, diable hdr exposure time
    blur_img = hdr2ldr(blur_data['colors'], 1)
    data = dict()
    data['blur'] = blur_img

    bproc.renderer.set_output_format("PNG")
    data.update(bproc.renderer.render_optical_flow(f'{output_dir}/tmp', f'{output_dir}/tmp', 
        get_backward_flow=True, get_forward_flow=True, blender_image_coordinate_style=False))

    bproc.writer.write_hdf5(f'{output_dir}/hdf5/slow', data)
    shutil.rmtree(f'{output_dir}/tmp')

    ########### second pass, render clean image for event simulation ########### 
    animation(output_dir, setup_info, {
        'animation_mode': config['animation_mode'],
        'num_frame': int(config['event_image_fps'] * config['duration']),
        'num_keyframes': config['num_keyframes'],
    })

    bproc.renderer.set_output_format("OPEN_EXR", 16)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 64
    bproc.renderer.set_cpu_threads(1)

    # bproc.renderer.enable_motion_blur(motion_blur_length=0.0)
    bpy.context.scene.render.use_motion_blur = False
    clean_data = bproc.renderer.render(f'{output_dir}/tmp')
    hdr_img = clean_data['colors']
    data = dict()
    data['hdr'] = hdr_img

    bproc.writer.write_hdf5(f'{output_dir}/hdf5/fast', data)
    shutil.rmtree(f'{output_dir}/tmp')



if __name__ == '__main__':
    main()





