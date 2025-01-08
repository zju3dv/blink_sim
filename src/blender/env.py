import blenderproc as bproc
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
import argparse, bpy, math, pickle, cv2, os, sys, shutil
from mathutils import Vector
import numpy as np
sys.path.insert(0, os.path.abspath(__file__+'/../../..'))
import json
import random
from pathlib import Path
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from src.blender.object_pool import (load_objects, load_texture_image,
                                           normalize_object, load_primitive,
                                           create_pyramid_for_camera)
from src.blender.tree_gen.tree import create_tree
from src.blender.animation import interpolate_trajectory
from src.blender.trajectory import (generate_trajectory,
                                          generate_trajectory_v2,
                                          random_pose_in_cam_frustum,
                                          compute_radius,
                                          euler_from_look_at,
                                          gen_camera_traj,
                                          normaliz_vec)
from src.blender.utils import retrieve_all
from src.blender.simulate_physics import simulate_physics_and_fix_final_poses


cfg = None

def sample_pose(obj: bproc.types.MeshObject):
    global cfg
    world_radius = cfg['world_radius']
    camera_active_radius = cfg['camera_active_radius']
    enough_height = cfg['enough_height']

    # Sample the spheres location above the surface
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))
    obj.set_location([0,0,0])
    bbox = obj.get_bound_box()
    min_z, max_z = min(bbox[:,2]), max(bbox[:,2])
    min_z = abs(min_z) if min_z < 0 else 0
    theta = np.random.uniform(-10/180*np.pi, np.pi+10/180*np.pi)
    radius = np.random.uniform(camera_active_radius, world_radius)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x = np.clip(x, a_min=-world_radius, a_max=world_radius)
    y = np.clip(y, a_min=-10, a_max=world_radius)
    min_height, max_height = 1+min_z, enough_height+min_z
    z = random.uniform(min_height, max_height)
    obj.set_location([x,y,z])


def setup_placement(config, use_custom_canopy):
    world_radius = config['world_radius']
    camera_active_radius = config['camera_active_radius']
    enough_height = config['enough_height']

    ground_plane = bproc.object.create_primitive(shape='PLANE')
    ground_plane.set_scale([world_radius,world_radius,1])
    ground_plane.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    ground_plane.set_location([0,0,0])

    objs_list = []
    num_static_obj = random.randint(*config['num_static_obj'])
    for i in range(num_static_obj):
        random_length = random.uniform(4.0, 15.0)
        random_height = random.uniform(4.0, 15.0)

        obj = load_primitive(config['wireframe_prob'])
        obj = normalize_object(obj)
        obj.set_scale([random_length, random_length, random_height])
        obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

        objs_list.append(obj)

    global cfg
    cfg = config
    bproc.object.sample_poses(
        objs_list,
        sample_pose,
        max_tries=200,
    )

    for obj in objs_list:
        obj.enable_rigidbody(True, mass=1)
        obj.blender_obj.rigid_body.restitution = 0.0
        obj.blender_obj.rigid_body.friction = 0.5

    ground_plane.enable_rigidbody(False)

    canopy_list = []
    _scale_list = [
        [world_radius,100,1],
        [world_radius,100,1],
        [100,world_radius,1],
        [100,world_radius,1],
    ]
    _loc_list = [
        [0,-10,int(enough_height/2)],
        [0,world_radius,int(enough_height/2)],
        [-world_radius,0,int(enough_height/2)],
        [world_radius,0,int(enough_height/2)],
    ]
    _rot_list = [
        [math.pi/2,0,0],
        [-math.pi/2,0,0],
        [0,-math.pi/2,0],
        [0,math.pi/2,0],
    ]
    for i in range(4):
        loc = _loc_list[i]
        canopy = bproc.object.create_primitive(shape='PLANE')
        canopy.set_scale(_scale_list[i])
        canopy.set_location(loc)
        canopy.set_rotation_euler(_rot_list[i])
        canopy.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
        canopy.enable_rigidbody(False)
        canopy_list.append(canopy)

    getattr(bpy.ops.mesh, f'primitive_cylinder_add')()
    cylinder = MeshObject(bpy.context.object)
    cylinder.set_scale([camera_active_radius, camera_active_radius, enough_height])
    cylinder.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    cylinder.enable_rigidbody(False)

    # Run the physics simulation
    simulate_physics_and_fix_final_poses(
        min_simulation_time=2,
        max_simulation_time=15,
        check_object_interval=1
    )

    # remove tmp objects
    bpy.data.objects.remove(ground_plane.blender_obj, do_unlink=True)
    bpy.data.objects.remove(cylinder.blender_obj, do_unlink=True)
    for canopy in canopy_list:
        bpy.data.objects.remove(canopy.blender_obj, do_unlink=True)

    plane_list = []
    # part_num = random.randint(1, 10)
    part_num = int(random.uniform(1, 3) ** 2)
    for i in range(part_num):
        for j in range(part_num):
            plane = bproc.object.create_primitive(shape='PLANE')
            length = math.ceil(world_radius/part_num)
            width = math.ceil(world_radius/part_num)
            plane.set_scale([length, width, 1])
            plane.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            y = (2*i+1-part_num) * length
            x = (2*j+1-part_num) * width
            plane.set_location([y,x,0])
            plane_list.append(plane)

    canopy_list = []
    if use_custom_canopy:
        canopy_height = 100
        _scale_list = [
            [math.ceil(world_radius/part_num), int(canopy_height/2),1],
            [math.ceil(world_radius/part_num),int(canopy_height/2),1],
            [int(canopy_height/2),math.ceil(world_radius/part_num),1],
            [int(canopy_height/2),math.ceil(world_radius/part_num),1],
        ]
        _loc_list = [
            [0,-world_radius,int(canopy_height/2)],
            [0,world_radius,int(canopy_height/2)],
            [-world_radius,0,int(canopy_height/2)],
            [world_radius,0,int(canopy_height/2)],
        ]
        _rot_list = [
            [math.pi/2,0,0],
            [-math.pi/2,0,0],
            [0,-math.pi/2,0],
            [0,math.pi/2,0],
        ]
        for i in range(4):
            for j in range(part_num):
                length = math.ceil(world_radius/part_num)
                y = (2*j+1-part_num) * length
                loc = _loc_list[i]
                loc = [y if math.fabs(item) < 1e-6 else item for item in loc]

                canopy = bproc.object.create_primitive(shape='PLANE')
                canopy.set_scale(_scale_list[i])
                canopy.set_location(loc)
                canopy.set_rotation_euler(_rot_list[i])
                canopy.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
                canopy_list.append(canopy)

    return plane_list, objs_list, canopy_list

def setup_material(config, ground, static_objs, canopys, mode):
    objs = ground + canopys + static_objs
    images = load_texture_image(config, mode, len(objs))
    for index, obj in enumerate(objs):
        if not obj.has_materials():
            obj.new_material("material_0")
        material_0 = obj.get_materials()[0]
        # add image texture as base color
        image = bpy.data.images.load(filepath=images[index])
        material_0.set_principled_shader_value("Base Color", image)
        # insert hsv node, to post-processing texture in hsv color space
        hsv_node = material_0.new_node(node_type="ShaderNodeHueSaturation")
        random_value = random.uniform(config['value_range'][0], config['value_range'][1])
        hsv_node.inputs["Value"].default_value = random_value

        src_node = material_0.get_the_one_node_with_type("TexImage")
        dst_node = material_0.get_the_one_node_with_type("BsdfPrincipled")

        random_metallic = random.uniform(config['metallic_range'][0], config['metallic_range'][1])
        dst_node.inputs['Metallic'].default_value = random_metallic
        random_specular = random.uniform(config['specular_range'][0], config['specular_range'][1])
        dst_node.inputs['Specular'].default_value = random_specular
        random_roughness = random.uniform(config['roughness_range'][0], config['roughness_range'][1])
        dst_node.inputs['Roughness'].default_value = random_roughness

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



def setup_camera_extrinsic(config):
    num_kf = config['num_keyframes']
    world_radius = config['world_radius']

    pos = np.array([0, 0, np.random.uniform(5, 15)])
    target = np.array([0, 50, np.random.uniform(0, 15)])
    # normalize target, then multiple by radius (value 40)
    target = normaliz_vec(target) * world_radius
    theta = np.random.uniform(math.pi/4, math.pi/4*3)
    up = np.array([math.cos(theta), 0, math.sin(theta)])
    euler = euler_from_look_at(pos, target, up)
    init_pose = [pos, target, euler]

    p_no_move = config['camera_static_ratio']
    if random.random() < p_no_move: 
        if config['animation_mode'] == 'linear':
            cam_pose_list = [init_pose] * num_kf
        elif config['animation_mode'] == 'cubic_spline':
            cam_pose_list = [init_pose] * (num_kf * 2 - 1)
    else:
        cam_pose_list = gen_camera_traj(init_pose, up, num_kf, config)

    return cam_pose_list


def setup_dynamic_objs(config, cam_pose_list, mode, static_obj_list):
    range_for_dynamic_obj = config['range_for_dynamic_obj']
    camera_active_radius = config['camera_active_radius']
    dynamic_objs = load_objects(config, mode)
    collision_check = config['collision_check']
    rotation_para = np.array(config['rotation_para']).astype(float)
    rotation_para[:,:2] = rotation_para[:,:2] / 180 * np.pi
    rotation_para[:,2] = rotation_para[:,2] / np.sum(rotation_para[:,2])

    for obj in dynamic_objs:
        obj = normalize_object(obj)
        obj.set_scale([random.uniform(3.0, 8.0)]*3)
        obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
        obj.set_origin(mode='CENTER_OF_MASS')

    # dynamic_objs, obj_pose_list = generate_trajectory(cam_pose_list,
    #                                 dynamic_objs, range_for_dynamic_obj)
    dynamic_objs, obj_pose_list = generate_trajectory_v2(cam_pose_list, dynamic_objs,
                                    range_for_dynamic_obj, camera_active_radius,
                                    static_obj_list, collision_check, rotation_para, animation_mode=config['animation_mode'])

    return dynamic_objs, obj_pose_list


def backup_alpha(all_obj_list):
    backup_alpha_table = {}
    backup_alpha_mute = {}

    all_obj_list = retrieve_all(all_obj_list)

    for obj in all_obj_list:
        blender_obj = obj.blender_obj
        if hasattr(blender_obj, 'data') and hasattr(blender_obj.data, 'materials'):
            for mat in blender_obj.data.materials:
                if mat is None or not hasattr(mat, 'name') or not hasattr(mat, 'use_nodes') or not hasattr(mat, 'node_tree'):
                    continue
                if mat.name in backup_alpha_table.keys() or mat.name in backup_alpha_mute.keys():
                    continue
                if not mat.use_nodes:
                    continue
                if not mat.node_tree:
                    continue
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                principled_bsdf = Utility.get_nodes_with_type(nodes, "BsdfPrincipled")
                if principled_bsdf:
                    for bsdf in principled_bsdf:
                        if bsdf.inputs['Alpha'].links:
                            backup_alpha_mute[mat.name] = bsdf.inputs['Alpha'].links[0].is_muted
                            if backup_alpha_mute[mat.name]:
                                backup_alpha_table[mat.name] = bsdf.inputs['Alpha'].default_value
                        else:
                            backup_alpha_table[mat.name] = bsdf.inputs['Alpha'].default_value

    return backup_alpha_table, backup_alpha_mute


def disable_alpha(all_obj_list):
    backup_alpha_table, backup_alpha_mute = backup_alpha(all_obj_list)

    all_obj_list = retrieve_all(all_obj_list)

    # to avoid incorrect depth for transparent objects, disable alpha texture
    for obj in all_obj_list:
        blender_obj = obj.blender_obj
        if hasattr(blender_obj, 'data') and hasattr(blender_obj.data, 'materials'):
            for mat in blender_obj.data.materials:
                if mat is None or not hasattr(mat, 'name') or not hasattr(mat, 'use_nodes') or not hasattr(mat, 'node_tree'):
                    continue
                if not mat.use_nodes:
                    continue
                if not mat.node_tree:
                    continue
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                principled_bsdf = Utility.get_nodes_with_type(nodes, "BsdfPrincipled")
                if principled_bsdf:
                    for bsdf in principled_bsdf:
                        if bsdf.inputs['Alpha'].links:
                            bsdf.inputs['Alpha'].links[0].is_muted = True
                            # links.remove(bsdf.inputs['Alpha'].links[0])
                        bsdf.inputs['Alpha'].default_value = 1
    
    return backup_alpha_table, backup_alpha_mute


def recover_alpha(all_obj_list, backup_alpha_table, backup_alpha_mute):
    all_obj_list = retrieve_all(all_obj_list)
    for obj in all_obj_list:
        blender_obj = obj.blender_obj
        if hasattr(blender_obj, 'data') and hasattr(blender_obj.data, 'materials'):
            for mat in blender_obj.data.materials:
                if mat is None or not hasattr(mat, 'name') or not hasattr(mat, 'use_nodes') or not hasattr(mat, 'node_tree'):
                    continue
                if not mat.use_nodes:
                    continue
                if not mat.node_tree:
                    continue
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                principled_bsdf = Utility.get_nodes_with_type(
                    nodes, "BsdfPrincipled")
                if principled_bsdf:
                    for bsdf in principled_bsdf:
                        if bsdf.inputs['Alpha'].links:
                            if mat.name in backup_alpha_mute:
                                bsdf.inputs['Alpha'].links[0].is_muted = backup_alpha_mute[mat.name]
                                if backup_alpha_mute[mat.name] and mat.name in backup_alpha_table:
                                    bsdf.inputs['Alpha'].default_value = backup_alpha_table[mat.name]
                        elif mat.name in backup_alpha_table:
                            bsdf.inputs['Alpha'].default_value = backup_alpha_table[mat.name]


def setup_camera_intrinsic(config):
    width, height = config['image_width'], config['image_height']
    bproc.camera.set_resolution(width, height)

    if 'hFoV' in config:
        hfov = config['hFoV'] / 180 * np.pi
        bproc.camera.set_intrinsics_from_blender_params(lens=hfov, lens_unit='FOV')

    if 'clip_start' in config:
        bproc.camera.set_intrinsics_from_blender_params(clip_start=config['clip_start'])

    if 'clip_end' in config:
        bproc.camera.set_intrinsics_from_blender_params(clip_end=config['clip_end'])

def setup_envmap(config):
    hdr_dir = config['hdr_dir']
    hdr_list = os.listdir(hdr_dir)
    hdr_name = random.choice(hdr_list)
    hdr_file = os.listdir(f'{hdr_dir}/{hdr_name}')[0]
    path = f'{hdr_dir}/{hdr_name}/{hdr_file}'
    bproc.world.set_world_background_hdr_img(path)

    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    mapping_node = nodes.get("Mapping")
    if mapping_node:
        mapping_node.inputs["Rotation"].default_value[2] = np.random.uniform(0, 2*np.pi)


def create_collection(name):

    # create collection
    create_collection = True
    for collection in bpy.data.collections:
        if collection.name == name:
            create_collection = False
            break
    if create_collection:
        main_collection = bpy.context.scene.collection
        new_collection = bpy.data.collections.new(name)
        main_collection.children.link(new_collection)


def setup_envmap_sphere_v2(config):
    hdr_dir = config['hdr_dir']
    hdr_list = os.listdir(hdr_dir)
    hdr_name = random.choice(hdr_list)
    hdr_file = os.listdir(f'{hdr_dir}/{hdr_name}')[0]
    path = f'{hdr_dir}/{hdr_name}/{hdr_file}'

    create_collection('hdr')
    # backup active collection
    active_collection = bpy.context.view_layer.active_layer_collection
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['hdr']

    # create a sphere
    sphere_radius = 1000 if 'envir_sphere_radius' not in config else config['envir_sphere_radius']
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=(0, 0, 0))
    sphere_obj = bpy.context.object

    # apply hdr mat
    hdr_mat = bpy.data.materials.new(name='hdr')
    sphere_obj.active_material = hdr_mat
    hdr_mat.use_nodes = True

    mat_output_node = hdr_mat.node_tree.nodes["Material Output"]
    environment_node = hdr_mat.node_tree.nodes.new("ShaderNodeTexEnvironment")
    emission_node = hdr_mat.node_tree.nodes.new("ShaderNodeEmission")
    transparent_node = hdr_mat.node_tree.nodes.new("ShaderNodeBsdfTransparent")
    light_path_node = hdr_mat.node_tree.nodes.new("ShaderNodeLightPath")
    mix_shader_node = hdr_mat.node_tree.nodes.new("ShaderNodeMixShader")

    hdr_mat.node_tree.links.new(environment_node.outputs['Color'], emission_node.inputs['Color'])
    hdr_mat.node_tree.links.new(emission_node.outputs['Emission'], mix_shader_node.inputs[2])
    hdr_mat.node_tree.links.new(transparent_node.outputs['BSDF'], mix_shader_node.inputs[1])
    hdr_mat.node_tree.links.new(light_path_node.outputs[0], mix_shader_node.inputs[0])
    hdr_mat.node_tree.links.new(mix_shader_node.outputs['Shader'], mat_output_node.inputs['Surface'])
    emission_node.inputs[1].default_value = 2

    hdr_image = bpy.data.images.load(path)
    environment_node.image = hdr_image

    # set world hdr
    world_obj = bpy.context.scene.world
    world_obj.use_nodes = True

    world_background_node = world_obj.node_tree.nodes["Background"]
    environment_node = world_obj.node_tree.nodes.new("ShaderNodeTexEnvironment")
    environment_node.image = hdr_image
    world_obj.node_tree.links.new(environment_node.outputs['Color'], world_background_node.inputs['Color'])
    world_background_node.inputs[1].default_value = 2.5

    # Restore active collection
    bpy.context.view_layer.active_layer_collection = active_collection


def setup_envmap_sphere(config):
    hdr_dir = config['hdr_dir']
    hdr_list = os.listdir(hdr_dir)
    hdr_name = random.choice(hdr_list)
    hdr_file = os.listdir(f'{hdr_dir}/{hdr_name}')[0]
    path = f'{hdr_dir}/{hdr_name}/{hdr_file}'

    # Create a large sphere
    sphere_radius = 1000 if 'envir_sphere_radius' not in config else config['envir_sphere_radius']
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=(0, 0, 0))
    sphere = bpy.context.active_object

    bpy.context.object.rotation_euler[0] = 1.5708
    bpy.context.object.rotation_euler[2] = np.random.uniform(0, 2*np.pi)

    # Correct the UV map for equirectangular projection
    bpy.ops.object.editmode_toggle()  # Enter edit mode
    bpy.ops.mesh.select_all(action='SELECT')  # Select all mesh parts
    bpy.ops.uv.sphere_project()  # Apply spherical projection
    bpy.ops.object.editmode_toggle()  # Return to object mode

    # Create a new material for the sphere
    mat = bpy.data.materials.new(name="HDR_Material")
    sphere.data.materials.append(mat)

    # Use nodes for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add emission shader
    emission_shader = nodes.new(type='ShaderNodeEmission')
    emission_shader.location = (0,0)

    # Add material output
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (300,0)

    # Add image texture (instead of environment texture)
    image_texture = nodes.new(type='ShaderNodeTexImage')
    image_texture.location = (-300,0)
    image_texture.image = bpy.data.images.load(path)  # Change to your image path

    # Add texture coordinate
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-600,0)

    # Connect the nodes
    mat.node_tree.links.new
    mat.node_tree.links.new(image_texture.inputs['Vector'], tex_coord.outputs['UV'])
    mat.node_tree.links.new(emission_shader.inputs['Color'], image_texture.outputs['Color'])
    mat.node_tree.links.new(material_output.inputs['Surface'], emission_shader.outputs['Emission'])

    # Optionally set strength of emission shader if necessary
    emission_shader.inputs["Strength"].default_value = 10  # Adjust this value as needed


def setup_tree(config, cam_pose_list, static_objs, animation_mode, collision_check,
               num_seg=50, tolerate_dist=3, num_attempt=30):
    tree_num = random.randint(1, 4)
    material_dir = config['tree_material_dir']
    init_pose = cam_pose_list[0]

    tree_list = []
    range_for_static_obj = config['range_for_static_obj']
    camera_active_radius = config['camera_active_radius']
    range_for_static_obj[2] = 0
    bvh_cache = {}
    for _ in range(tree_num):
        xy_scale = [random.uniform(5.0, 10.0)] * 2
        z_scale = random.uniform(5.0, 10.0)
        for _ in range(num_attempt):
            success = True
            pos = random_pose_in_cam_frustum(init_pose[0], init_pose[1], range_for_static_obj)
            pos[0] = np.clip(pos[0], a_min=-camera_active_radius, a_max=camera_active_radius)
            pos[1] = np.clip(pos[1], a_min=0, a_max=camera_active_radius)
            pos[2] = 0
            cube = bproc.object.create_primitive(shape='CUBE')
            cube.set_scale(xy_scale + [z_scale])
            cube.set_location(pos[:2].tolist() + [z_scale/2])

            K_matrix = bproc.camera.get_intrinsics_as_K_matrix()
            resolution = (bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x)
            camera_mesh = create_pyramid_for_camera(K_matrix, resolution, tolerate_dist)
            pos_list = [pos for (pos, lookat, euler) in cam_pose_list]
            euler_list = [euler for (pos, lookat, euler) in cam_pose_list]
            num_kf = len(pos_list)
            loc, rot = interpolate_trajectory(num_seg*num_kf, num_kf, pos_list, euler_list, animation_mode=animation_mode)
            for k in range(len(loc)):
                camera_mesh.set_location(loc[k])
                camera_mesh.set_rotation_euler(rot[k])
                if camera_mesh.get_name() in bvh_cache:
                    del bvh_cache[camera_mesh.get_name()]
                no_collision = CollisionUtility.check_intersections(
                    cube, bvh_cache, static_objs+[camera_mesh], static_objs+[camera_mesh])
                if collision_check and not no_collision:
                    success = False
                    break
            cube.delete()
            camera_mesh.delete()

            if not success:
                continue

            tree = create_tree(material_dir)
            tree = normalize_object(tree)

            tree.set_scale(xy_scale + [z_scale])
            tree.move_origin_to_bottom_mean_point()
            tree.set_location(pos)
            tree.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            tree_list.append(tree)
            break
    
    return tree_list

def check_cam_distance(cam_pose_list, static_objs, animation_mode, num_seg=50, tolerate_dist=3):
    valid_list = [True] * len(static_objs)
    bvh_cache = {}

    K_matrix = bproc.camera.get_intrinsics_as_K_matrix()
    resolution = (bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x)
    camera_mesh = create_pyramid_for_camera(K_matrix, resolution, tolerate_dist)
    pos_list = [pos for (pos, lookat, euler) in cam_pose_list]
    euler_list = [euler for (pos, lookat, euler) in cam_pose_list]
    num_kf = len(pos_list)
    loc, rot = interpolate_trajectory(num_seg*num_kf, num_kf, pos_list, euler_list, animation_mode=animation_mode)
    for k in range(len(loc)):
        camera_mesh.set_location(loc[k])
        camera_mesh.set_rotation_euler(rot[k])
        if camera_mesh.get_name() in bvh_cache:
            del bvh_cache[camera_mesh.get_name()]
        for i, obj in enumerate(static_objs):
            intersection, bvh_cache = CollisionUtility.check_mesh_intersection(
                camera_mesh, obj, bvh_cache=bvh_cache, skip_inside_check=True)
            if intersection:
                valid_list[i] = False
    camera_mesh.delete()
    new_obj_list = []
    for i in range(len(valid_list)):
        if not valid_list[i]:
            static_objs[i].delete()
            continue
        new_obj_list.append(static_objs[i])

    return new_obj_list



def setup_env(config, mode):
    setup_camera_intrinsic(config)
    outdoor = random.random() < config['outdoor_ratio']
    use_custom_canopy = config['canopy_type'] == 'cube'
    cam_pose_list = setup_camera_extrinsic(config)
    ground, static_objs, canopys = setup_placement(config, use_custom_canopy)
    setup_material(config, ground, static_objs, canopys, mode)
    static_objs = check_cam_distance(cam_pose_list, static_objs, config['animation_mode'])
    if outdoor:
        trees = setup_tree(config, cam_pose_list, static_objs, config['animation_mode'], config['collision_check'])
        static_objs = static_objs + trees
    # setup_lighting(cam_pose_list[0])
    if use_custom_canopy:
        setup_envmap(config)
    else:
        setup_envmap_sphere_v2(config)
    pos_euler_list = [[pos, euler] for (pos, lookat, euler) in cam_pose_list]
    dynamic_objs, obj_pose_list = setup_dynamic_objs(config, cam_pose_list, mode, static_objs)

    all_obj_list = static_objs + dynamic_objs
    if outdoor:
        all_obj_list += trees
    setup_info = {
        'cam_pose': pos_euler_list,
        'dynamic_objs': dynamic_objs,
        'dyna_objs_pose': obj_pose_list,
        'all_obj_list': all_obj_list,
    }

    return setup_info

def set_engines(engine_type, samples, long_animation=False, denoise=True, cpu_threads=4):
    bproc.renderer.set_cpu_threads(cpu_threads)

    # optimized options for rendering long-time animation
    if long_animation:
        bpy.context.scene.render.use_persistent_data = True

    # WARNING: the following setting do not affect the actual rendering result, they are only "display" or "view" setting
    # if raw_color:
    #     bpy.context.scene.display_settings.display_device = 'None'
    #     bpy.context.scene.view_settings.view_transform = 'Standard'
    #     bpy.context.scene.view_settings.look = 'None'
    # else:
    #     bpy.context.scene.display_settings.display_device = 'sRGB'
    #     bpy.context.scene.view_settings.view_transform = 'Filmic'
    #     bpy.context.scene.view_settings.look = 'High Contrast'

    if engine_type == 'cycles':
        bpy.context.scene.render.engine = 'CYCLES'
        bproc.renderer.set_max_amount_of_samples(samples)
        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'
        if denoise:
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
            bpy.context.scene.cycles.denoising_prefilter = 'FAST'
        else:
            bpy.context.scene.cycles.use_denoising = False
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = samples
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_bloom = True


def setup_settings_config(settings_config, settings_attrs=[]):
    for settings_attr in settings_attrs:
        settings_field = getattr(bpy.context.scene, settings_attr)
        for attr, value in settings_config[settings_attr].items():
            if hasattr(settings_field, attr):
                try:
                    setattr(settings_field, attr, value)
                except:
                    pass


def backup_keyframe(obj_list):
    obj_keyframe = {}
    for obj in obj_list:
        blender_obj = obj.blender_obj
        curves = blender_obj.animation_data.action.fcurves
        sorted_curves = sorted(curves, key=lambda fc: (fc.data_path, fc.array_index))
        obj_keyframe[blender_obj.name] = [[] for _ in range(len(sorted_curves))]
        for i, c in enumerate(sorted_curves):
            keyframes = c.keyframe_points
            pts = [(kf.co.x, kf.co.y) for kf in keyframes]
            obj_keyframe[blender_obj.name][i] = pts

    data = {
        'frame_start': bpy.context.scene.frame_start,
        'frame_end': bpy.context.scene.frame_end,
        'obj_keyframe': obj_keyframe,
    }

    return data


def restore_keyframe(obj_list, data):
    bpy.context.scene.frame_start = data['frame_start']
    bpy.context.scene.frame_end = data['frame_end']
    obj_keyframe = data['obj_keyframe']

    for obj in obj_list:
        blender_obj = obj.blender_obj
        curves = blender_obj.animation_data.action.fcurves
        sorted_curves = sorted(curves, key=lambda fc: (fc.data_path, fc.array_index))
        backup_curves = obj_keyframe[blender_obj.name]
        for i, c in enumerate(sorted_curves):
            c.keyframe_points.clear()
            for (x, y) in backup_curves[i]:
                c.keyframe_points.insert(x, y, options={'NEEDED', 'FAST'})
            c.update()


def rescale_keyframe(obj_list, time_table=None):
    min_x = min(time_table.keys())
    max_x = max(time_table.keys())
    min_y = time_table[min_x]
    max_y = time_table[max_x]
    bpy.context.scene.frame_end = int(max_y + 1)
    for obj in obj_list:
        blender_obj = obj.blender_obj
        curves = blender_obj.animation_data.action.fcurves
        for c in curves:
            pts = []
            for ts in range(min_x, max_x+1):
                value = c.evaluate(ts)
                pts.append((ts, value))
            c.keyframe_points.clear()
            for (ts, value) in pts:
                c.keyframe_points.insert(time_table[ts], value, options={'NEEDED', 'FAST'})
            c.update()

def analyze_camera_movement(camera, start_frame, end_frame):
    # Save the current frame
    current_frame = bpy.context.scene.frame_current

    pos_distances = []
    rot_distances = []

    prev_pos = None
    prev_rot = None

    for frame_num in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame_num)

        pos = camera.location
        rot = camera.rotation_euler

        if prev_pos and prev_rot:
            pos_distance = [abs(d) for d in pos - prev_pos]
            rot_distance = [abs(r1-r2) for r1,r2 in zip(rot, prev_rot)]
            pos_distances.append(pos_distance)
            rot_distances.append(rot_distance)

        prev_pos = pos.copy()
        prev_rot = rot.copy()

    avg_pos_distance = [sum(d) / len(d) for d in zip(*pos_distances)]
    avg_rot_distance = [sum(r) / len(r) for r in zip(*rot_distances)]

    bpy.context.scene.frame_set(current_frame)

    return avg_pos_distance, avg_rot_distance

def switch_to_left_camera(original_cam_pos):
    current_frame = bpy.context.scene.frame_current
    cam_obj = bpy.context.scene.camera
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    for frame_index, i in zip(range(frame_start, frame_end), range(len(original_cam_pos))):
        bpy.context.scene.frame_set(frame_index)

        cam_obj.location = original_cam_pos[i]
        cam_obj.keyframe_insert(data_path="location", frame=frame_index)

    # Restore the original frame
    bpy.context.scene.frame_set(current_frame)

def switch_to_right_camera(baseline_length):
    original_cam_pos = []

    current_frame = bpy.context.scene.frame_current
    cam_obj = bpy.context.scene.camera
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    for i in range(frame_start, frame_end):
        bpy.context.scene.frame_set(i)

        cam_obj.keyframe_insert(data_path="location", frame=i)

    for i in range(frame_start, frame_end):
        bpy.context.scene.frame_set(i)

        pos = cam_obj.location
        original_cam_pos.append(pos.copy())

        # Determine the right-shift in the camera's local coordinates
        # import ipdb; ipdb.set_trace()
        right_shift = cam_obj.matrix_world.to_3x3() @ Vector((baseline_length, 0.0, 0.0))

        # Update the camera's position
        cam_obj.location = pos + right_shift
        cam_obj.keyframe_insert(data_path="location", frame=i)

    # Restore the original frame
    bpy.context.scene.frame_set(current_frame)

    return original_cam_pos


def add_camera_jitter(config):
    if 'enable_camera_jittor' in config and not config['enable_camera_jittor']:
        return

    attenuate = 0.2
    if 'camera_jittor_attenuate' in config:
        attenuate = config['camera_jittor_attenuate']

    # Save the current frame
    current_frame = bpy.context.scene.frame_current

    cam_obj = bpy.context.scene.camera
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    avg_pos_distance, avg_rot_distance = analyze_camera_movement(cam_obj, frame_start, frame_end)
    avg_pos_distance = [j*attenuate for j in avg_pos_distance]
    avg_rot_distance = [j*attenuate for j in avg_rot_distance]

    for i in range(frame_start, frame_end):
        bpy.context.scene.frame_set(i)

        pos = cam_obj.location
        rot = cam_obj.rotation_euler

        jitter_pos = [random.gauss(0, j) for j in avg_pos_distance]
        new_pos = [p + j for p, j in zip(pos, jitter_pos)]

        jitter_rot = [random.gauss(0, j) for j in avg_rot_distance]
        new_rot = [r + j for r, j in zip(rot, jitter_rot)]

        cam_obj.location = new_pos
        cam_obj.rotation_euler = new_rot

        cam_obj.keyframe_insert(data_path="location", frame=i)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=i)

    # Restore the original frame
    bpy.context.scene.frame_set(current_frame)


def set_defocus_params(enable, distance=10):
    bpy.context.scene.camera.data.dof.use_dof = enable
    bpy.context.scene.camera.data.dof.focus_distance = distance


def setup_atmospheric_effects(config):
    prob = config['atmospheric_effects_prob']
    if np.random.rand() > prob:
        return

    density_range = config['atmospheric_effects_density_range']
    strength_range = config['atmospheric_effects_strength_range']

    getattr(bpy.ops.mesh, f'primitive_cube_add')()
    obj = MeshObject(bpy.context.object)
    obj.blender_obj.name = "atmospheric_effects_cube"
    scale = config['world_radius'] * 2
    obj.set_scale([scale, scale, scale])

    obj.new_material("vol")
    material_0 = obj.get_materials()[0]
    vol_node = material_0.new_node(node_type="ShaderNodeVolumePrincipled")
    vol_node.inputs["Density"].default_value = np.random.uniform(*density_range)
    vol_node.inputs["Emission Strength"].default_value = np.random.uniform(*strength_range)

    dst_node = material_0.get_the_one_node_with_type("OutputMaterial")

    Utility.get_node_connected_to_the_output_and_unlink_it(material_0.blender_obj)

    links = material_0.blender_obj.node_tree.links
    links.new(vol_node.outputs["Volume"], dst_node.inputs["Volume"])



def remove_atmospheric_effects():
    # find the cube
    for obj in bpy.data.objects:
        if obj.name == "atmospheric_effects_cube":
            obj.select_set(True)
            bpy.ops.object.delete()
            break