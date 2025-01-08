import blenderproc as bproc
import bpy
import json
import os
import numpy as np
import random
from pathlib import Path
from typing import List
from blenderproc.python.loader.CCMaterialLoader import _CCMaterialLoader
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.utility.Utility import Utility, resolve_path
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.types.MeshObjectUtility import MeshObject
import math
from src.utils import safe_sample

path_dict = {
    'shapenet': 'ShapeNetCore.v2',
    'google_scanned': 'google_scanned',
    'primitive': None,
    'ade20k': 'ADE20K',
    'cc_textures_diffuse': 'cc_textures_diffuse',
    'flickr': 'flickr',
    'pixabay': 'pixabay',
}

def normalize_object(obj):
    max_axis_len = max(obj.get_bound_box().max(axis=0) - obj.get_bound_box().min(axis=0))

    if math.fabs(max_axis_len) < 1e-6:
        return obj

    scale_ratio = 1 / max_axis_len
    obj.set_scale([scale_ratio]*3)
    obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    return obj


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

def parse_mtl(filename):
    materials = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    current_material = None
    for line in lines:
        split_line = line.strip().split()
        if not split_line:
            continue

        prefix = split_line[0]
        args = split_line[1:]

        if prefix == 'newmtl':
            current_material = args[0]
            materials[current_material] = {}
        elif current_material is not None:
            if prefix in ['Ka', 'Kd', 'Ks', 'Tf', 'illum', 'd', 'Ns', 'sharpness', 'Ni']:
                materials[current_material][prefix] = list(map(float, args))
            elif prefix in ['map_Ka', 'map_Kd', 'map_Ks', 'map_Ns', 'map_d', 'disp', 'decal', 'bump']:
                materials[current_material][prefix] = args[0]

    return materials

def write_mtl(filename, materials):
    with open(filename, 'w') as file:
        for material, properties in materials.items():
            file.write(f'newmtl {material}\n')
            for property, value in properties.items():
                if isinstance(value, list):
                    file.write(f'{property} {" ".join(map(str, value))}\n')
                else:
                    file.write(f'{property} {value}\n')
            file.write('\n')

def load_shapenet(filepath, config, mode):
    if np.random.rand() < config['augment_shapenet_prob']:
        mtl_path = filepath.replace('.obj', '.mtl')
        material = parse_mtl(mtl_path)

        key_num = len(material.keys())
        image_list = load_texture_image(config, mode, key_num)

        for i, key in enumerate(material.keys()):
            # random texture
            texture_path = image_list[i]
            material[key]['map_Kd'] = f'{texture_path}'
            # disable transparent
            if 'd' in material[key].keys():
                material[key]['d'] = 1.0

        # copy the original folder
        folder = os.path.dirname(filepath)
        obj_name = os.path.basename(filepath)
        if folder.endswith('/'):
            folder = folder[:-1]
        os.system(f'cp -r {folder} {folder}_bak')
        folder = f'{folder}_bak'
        filepath = f'{folder}/{obj_name}'
        mtl_path = filepath.replace('.obj', '.mtl')
        
        # override the new mtl
        write_mtl(mtl_path, material)

        loaded_objects = bproc.loader.load_obj(filepath, use_legacy_obj_import=True)

        # delete the new folder
        os.system(f'rm -rf {folder}')
    else:
        loaded_objects = bproc.loader.load_obj(filepath, use_legacy_obj_import=True)

    obj = loaded_objects[0]
    correct_materials(obj)

    # removes the x axis rotation found in all ShapeNet objects, this is caused by importing .obj files
    # the object has the same pose as before, just that the rotation_euler is now [0, 0, 0]
    obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)

    bpy.ops.object.select_all(action='DESELECT')

    return obj

def load_google_scanned(filepath, config, mode):
    loaded_objects = bproc.loader.load_obj(filepath, use_legacy_obj_import=True)

    obj = loaded_objects[0]

    correct_materials(obj)

    bpy.ops.object.select_all(action='DESELECT')

    return obj

def load_primitive(wireframe_prob):
    shape_list = ['cube', 'uv_sphere', 'cylinder', 'ico_sphere', 'cone', 'torus']
    shape_prob_for_wireframe = {
        'cube': 0.3,
        'uv_sphere': 0.1,
        'cylinder': 0.3,
        'ico_sphere': 0.1,
        'cone': 0.1,
        'torus': 0.1,
    }
    thickness_table = {
        'cube': [[0.2, 0.4], [0.4, 0.7], [0.7, 1.0]],
        'uv_sphere': [[0.035, 0.05], [0.05, 0.07]],
        'cylinder': [[0.07, 0.10], [0.10, 0.12], [0.12, 0.15]],
        'ico_sphere': [[0.07, 0.10], [0.10, 0.12], [0.12, 0.15]],
        'cone': [[0.06, 0.08], [0.08, 0.1]],
        'torus': [[0.03, 0.05], [0.05, 0.07]],
    }

    if np.random.rand() < wireframe_prob:
        shape_index = np.random.choice(len(shape_list), p=[shape_prob_for_wireframe[shape] for shape in shape_list])
        shape = shape_list[shape_index]
        getattr(bpy.ops.mesh, f'primitive_{shape}_add')()
        obj = MeshObject(bpy.context.object)

        thickness_setting = thickness_table[shape]
        thickness_range = random.choice(thickness_setting)
        thickness = random.uniform(thickness_range[0], thickness_range[1])
        obj.add_modifier('WIREFRAME', thickness=thickness)
        bpy.ops.object.modifier_apply({"object": obj.blender_obj}, modifier="Wireframe")

    else:
        shape = random.choice(shape_list)
        getattr(bpy.ops.mesh, f'primitive_{shape}_add')()
        obj = MeshObject(bpy.context.object)

    return obj



def sample_class_num(prop, total_num):
    class_num = [0] * len(prop)
    for i in range(total_num):
        class_num[np.random.choice(len(prop), p=prop)] += 1
    return class_num


def load_objects(config, mode):
    shape_num = random.randrange(config['shape_num'][0], config['shape_num'][1])
    class_num = sample_class_num(config['object_pool_prob'], shape_num)
    name_list = config['object_pool']
    obj_list = []
    for i, num in enumerate(class_num):
        name = name_list[i]
        if path_dict[name] is not None:
            data_dir = Path(f'data/{path_dict[name]}')
            assert data_dir.is_dir(), f'{data_dir} is not a directory'
            if config.get('example_assets', False):
                split_file = f'configs/example/{name}_example.json'
            else:
                split_file = f'configs/{name}.json'

            with open(split_file, 'r') as fr:
                rel_path = json.load(fr)[mode]
                all_shapes_p = [data_dir.joinpath(rp) for rp in rel_path]

            shapes_p = safe_sample(all_shapes_p, num)
            shapes_p = [shape_p for shape_p in shapes_p if shape_p.is_file()]

            objs = [globals()[f'load_{name}'](str(shape_p), config, mode) for shape_p in shapes_p]
        else:
            wireframe_prob = 1.0
            objs = [globals()[f'load_{name}'](wireframe_prob) for _ in range(num)]

        obj_list += objs

    return obj_list


def load_texture_image(config, mode, number):
    class_num = sample_class_num(config['texture_pool_prob'], number)
    name_list = config['texture_pool']
    image_list = []
    for i, num in enumerate(class_num):
        name = name_list[i]
        data_dir = Path(f'data/{path_dict[name]}')
        assert data_dir.is_dir(), f'{data_dir} is not a directory'
        if config.get('example_assets', False):
            split_file = f'configs/example/{name}_example.json'
        else:
            split_file = f'configs/{name}.json'

        with open(split_file, 'r') as fr:
            rel_path = json.load(fr)[mode]
            all_images = [str(data_dir.joinpath(rp)) for rp in rel_path]

        images = safe_sample(all_images, num)
        images = [img for img in images if os.path.exists(img)]
        image_list += images

    random.shuffle(image_list)

    return image_list


def load_if_exist_fake_mat(path_dict):
    fake_base_image_path = path_dict['base_image_path'].replace("Color", "FakeColor")
    if os.path.exists(fake_base_image_path):
        path_dict['base_image_path'] = path_dict['base_image_path'].replace("Color", "FakeColor")
        path_dict['alpha_image_path'] = path_dict['alpha_image_path'].replace("Opacity", "FakeOpacity")
        path_dict['ambient_occlusion_image_path'] = path_dict['ambient_occlusion_image_path'].replace("AmbientOcclusion", "FakeAmbientOcclusion")
        path_dict['metallic_image_path'] = path_dict['metallic_image_path'].replace("Metalness", "FakeMetalness")
        path_dict['roughness_image_path'] = path_dict['roughness_image_path'].replace("Roughness", "FakeRoughness")
        path_dict['normal_image_path'] = path_dict['normal_image_path'].replace("Normal", "FakeNormal")
        path_dict['displacement_image_path'] = path_dict['displacement_image_path'].replace("Displacement", "FakeDisplacement")
    
    return path_dict



def load_ccmaterials(folder_path, used_assets, preload = False,
                     fill_used_empty_materials = False, add_custom_properties = None,
                     load_cnt = 0, given_hue=None):

    folder_path = resolve_path(folder_path)

    if used_assets is not None:
        used_assets = [asset.lower() for asset in used_assets]

    if add_custom_properties is None:
        add_custom_properties = {}

    if preload and fill_used_empty_materials:
        raise Exception("Preload and fill used empty materials can not be done at the same time, check config!")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        materials = []
        asset_path = os.listdir(folder_path)
        random.shuffle(asset_path)
        for asset in asset_path:
            if used_assets:
                skip_this_one = True
                for used_asset in used_assets:
                    # lower is necessary here, as all used assets are made that that way
                    if asset.lower().startswith(used_asset.replace(" ", "")):
                        skip_this_one = False
                        break
                if skip_this_one:
                    continue
            current_path = os.path.join(folder_path, asset)
            if os.path.isdir(current_path):
                path_dict = {}
                base_image_path = [_f for _f in os.listdir(current_path) if '_Color' in _f][0]
                path_dict['base_image_path'] = f'{current_path}/{base_image_path}'

                path_dict['ambient_occlusion_image_path'] = path_dict['base_image_path'].replace("Color", "AmbientOcclusion")
                path_dict['metallic_image_path'] = path_dict['base_image_path'].replace("Color", "Metalness")
                path_dict['roughness_image_path'] = path_dict['base_image_path'].replace("Color", "Roughness")
                path_dict['alpha_image_path'] = path_dict['base_image_path'].replace("Color", "Opacity")
                path_dict['normal_image_path'] = path_dict['base_image_path'].replace("Color", "Normal")
                if not os.path.exists(path_dict['normal_image_path']):
                    path_dict['normal_image_path'] = path_dict['base_image_path'].replace("Color", "NormalDX")
                path_dict['displacement_image_path'] = path_dict['base_image_path'].replace("Color", "Displacement")

                load_if_exist_fake_mat(path_dict)

                # if the material was already created it only has to be searched
                if fill_used_empty_materials:
                    new_mat = MaterialLoaderUtility.find_cc_material_by_name(asset, add_custom_properties)
                else:
                    new_mat = MaterialLoaderUtility.create_new_cc_material(asset, add_custom_properties)

                # if preload then the material is only created but not filled
                if preload:
                    # Set alpha to 0 if the material has an alpha texture, so it can be detected
                    # e.q. in the material getter.
                    nodes = new_mat.node_tree.nodes
                    principled_bsdf = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
                    principled_bsdf.inputs["Alpha"].default_value = 0 if os.path.exists(path_dict['alpha_image_path']) else 1
                    # add it here for the preload case
                    materials.append(Material(new_mat))
                    continue
                if fill_used_empty_materials and not MaterialLoaderUtility.is_material_used(new_mat):
                    # now only the materials, which have been used should be filled
                    continue

                # create material based on these image paths
                _CCMaterialLoader.create_material(new_mat, path_dict['base_image_path'], path_dict['ambient_occlusion_image_path'],
                                                  path_dict['metallic_image_path'], path_dict['roughness_image_path'],
                                                  path_dict['alpha_image_path'], path_dict['normal_image_path'],
                                                  path_dict['displacement_image_path'])
                new_mat.blend_method = 'CLIP'

                mat_bproc = Material(new_mat)

                if given_hue is not None:
                    hsv_node = mat_bproc.new_node(node_type="ShaderNodeHueSaturation")
                    hsv_node.inputs["Hue"].default_value = given_hue
                    dst_node = mat_bproc.get_the_one_node_with_type("BsdfPrincipled")
                    src_node = None
                    for link in mat_bproc.links:
                        if link.to_socket == dst_node.inputs["Base Color"]:
                            src_node = link.from_node
                            break
                    Utility.insert_node_instead_existing_link(
                        mat_bproc.links,
                        src_node.outputs["Color"],
                        hsv_node.inputs["Color"],
                        hsv_node.outputs["Color"],
                        dst_node.inputs["Base Color"]
                    )

                materials.append(mat_bproc)
                if len(materials) >= load_cnt:
                    break
        return materials
    raise FileNotFoundError(f"The folder path does not exist: {folder_path}")


def create_pyramid_for_camera(K, resolution, depth=3):
    # Intrinsic Parameters from the Matrix
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    image_height, image_width = resolution

    # Compute pyramid (frustum) corners
    x_left = depth * (0 -cx) / fx
    x_right = depth * (image_width - cx) / fx  # Assuming image width normalized to [0, 1]
    y_top = depth * (0 -cy) / fy
    y_bottom = depth * (image_height - cy) / fy  # Assuming image height normalized to [0, 1]

    top_left = (x_left, y_top, -depth)
    top_right = (x_right, y_top, -depth)
    bottom_left = (x_left, y_bottom, -depth)
    bottom_right = (x_right, y_bottom, -depth)

    # Create pyramid mesh
    verts = [(0, 0, 0), top_left, top_right, bottom_left, bottom_right]
    edges = []
    faces = [(0, 1, 2), (0, 2, 4), (0, 4, 3), (0, 3, 1), (1, 3, 4, 2)]
    
    mesh = bpy.data.meshes.new(name="PyramidMesh")
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    
    pyramid = bpy.data.objects.new("Pyramid", mesh)
    bpy.context.collection.objects.link(pyramid)

    pyramid_obj = MeshObject(pyramid)

    return pyramid_obj
