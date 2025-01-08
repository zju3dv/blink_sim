import blenderproc as bproc
import os
import bpy
import bmesh
import numpy as np
from typing import Optional
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.modules.main.GlobalStorage import GlobalStorage
import random

def get_vertices_xyz_v2(obj, face_id):
    mesh = obj.data

    xyz = np.zeros((3, 3))
    if face_id >= 0 and face_id < len(mesh.polygons):
        poly = mesh.polygons[face_id]
        for i, loop_index in enumerate(poly.loop_indices):
            # Those faces that aren't triangulated successfully, it possibly won't pass the UV check and depth check.
            if i >= 3:
                break
            loop = mesh.loops[loop_index]
            vertex_index = loop.vertex_index
            vertex = mesh.vertices[vertex_index]
            xyz[i] = vertex.co
    else:
        # set to very far place to set NO_CAST status
        xyz[:] = -1000000

    return xyz



def get_vertices_xyz(obj_name, face_id, bmesh_obj=None):

    if bmesh_obj is None:
        ob = bpy.data.objects[obj_name]
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(ob, depsgraph)
    else:
        bm = bmesh_obj

    xyz = np.zeros((3, 3))

    if face_id < len(bm.faces):
        bm.faces.ensure_lookup_table()
        face = bm.faces[face_id]
        xyz[0] = face.verts[0].co
        xyz[1] = face.verts[1].co
        xyz[2] = face.verts[2].co
    else:
        # set to very far place to set NO_CAST status
        xyz = -1000000

    return xyz



def enable_uv_output(output_dir: Optional[str] = None, file_prefix: str = "uv_", output_key: str = "uv"):
    """ Enables writing uv color (albedo) images.
    uv color images will be written in the form of .png files during the next rendering.
    :param output_dir: The directory to write files to, if this is None the temporary directory is used.
    :param file_prefix: The prefix to use for writing the files.
    :param output_key: The key to use for registering the uv color output.
    """
    if output_dir is None:
        output_dir = Utility.get_temporary_directory()

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    bpy.context.view_layer.use_pass_uv = True
    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, 'CompositorNodeRLayers')
    final_output = render_layer_node.outputs["UV"]

    output_file = tree.nodes.new('CompositorNodeOutputFile')
    output_file.base_path = output_dir
    output_file.format.file_format = "OPEN_EXR"
    output_file.file_slots.values()[0].path = file_prefix
    links.new(final_output, output_file.inputs['Image'])

    Utility.add_output_entry({
        "key": output_key,
        "path": os.path.join(output_dir, file_prefix) + "%04d" + ".exr",
        "version": "2.0.0"
    })


def backup_uv(objs):
    backup_table = {}

    obj_list = retrieve_all(objs)

    blender_obj_list = [obj.blender_obj for obj in obj_list]

    for blender_obj in blender_obj_list:
        if blender_obj.type != 'MESH':
            continue
        mesh = blender_obj.data
        if blender_obj.name in backup_table.keys():
            continue
        for uv_layer in mesh.uv_layers:
            if uv_layer.active_render:
                backup_table[blender_obj.name] = uv_layer.name
                break

    return backup_table

def cleanup_mesh(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode='EDIT')  # Switch to Edit mode
    bpy.ops.mesh.select_all(action='SELECT')  # Select all vertices

    bpy.ops.mesh.remove_doubles()  # Remove duplicate vertices

    # bpy.ops.mesh.select_all(action='DESELECT')  # Deselect all vertices
    # bpy.ops.mesh.select_non_manifold()  # Select non-manifold vertices
    # bpy.ops.mesh.delete(type='EDGE')

    bpy.ops.mesh.normals_make_consistent(inside=False)  # Recalculate normals outside
    
    bpy.ops.object.mode_set(mode='OBJECT')  # Back to Object mode

def check_object_exists(name):
    return bpy.data.objects.get(name) is not None

def retrieve_all(objs):
    obj_list = [obj for obj in objs if check_object_exists(obj.get_name())]
    for obj in objs:
        if hasattr(obj, 'get_children'):
            childs = obj.get_children(return_all_offspring=True)
            obj_list += childs

    # for obj in obj list, keep only one instance if multiple elements have the same name
    obj_list_curr = []
    obj_names = set()
    for obj in obj_list:
        if obj.blender_obj.name not in obj_names:
            obj_list_curr.append(obj)
            obj_names.add(obj.blender_obj.name)
    obj_list = obj_list_curr

    return obj_list


def set_faceID_uv(objs):
    backup_table = backup_uv(objs)

    obj_list = retrieve_all(objs)

    for obj in obj_list:
        blender_obj = obj.blender_obj
        if blender_obj.type == 'MESH':

            mesh = blender_obj.data
            uvs = [0.0 for _ in range(2 * len(mesh.loops))]
            for poly in mesh.polygons:
                # TODO: we dont't know why it works and when it doesn't work
                pseudo_u = int(poly.index / 255)
                pseudo_v = int(poly.index % 255)
                for loop_idx in poly.loop_indices:
                    uvs[2 * loop_idx] = pseudo_u
                    uvs[2 * loop_idx + 1] = pseudo_v

            if 'FaceIndex' not in mesh.uv_layers:
                mesh.uv_layers.new(name='FaceIndex')

            mesh.uv_layers['FaceIndex'].data.foreach_set('uv', uvs)
            mesh.uv_layers['FaceIndex'].active = True
            mesh.uv_layers['FaceIndex'].active_render = True

    return backup_table

def set_barycentric_coord_uv(objs):
    backup_table = backup_uv(objs)

    obj_list = retrieve_all(objs)

    for obj in obj_list:
        blender_obj = obj.blender_obj
        if blender_obj.type == 'MESH':
            mesh = blender_obj.data
            uvs = [0.0 for _ in range(2 * len(mesh.loops))]
            for poly in mesh.polygons:
                for i, loop_idx in enumerate(poly.loop_indices):
                    if i == 0:
                        uvs[2 * loop_idx] = 1 # (1, 0)
                    elif i == 1:
                        uvs[2 * loop_idx + 1] = 1 # (0, 1)

            if 'BarycentricCoord' not in mesh.uv_layers:
                mesh.uv_layers.new(name='BarycentricCoord')

            mesh.uv_layers['BarycentricCoord'].data.foreach_set('uv', uvs)
            mesh.uv_layers['BarycentricCoord'].active = True
            mesh.uv_layers['BarycentricCoord'].active_render = True

    return backup_table


def unset_faceID_uv(objs, backup_table):
    obj_list = retrieve_all(objs)

    for obj in obj_list:
        blender_obj = obj.blender_obj
        if blender_obj.type == 'MESH':
            mesh = blender_obj.data
            remove_uv = mesh.uv_layers.get('FaceIndex')
            if remove_uv is not None:
                mesh.uv_layers.remove(remove_uv)
            if blender_obj.name in backup_table:
                origin_uv_name = backup_table[blender_obj.name]
                mesh.uv_layers[origin_uv_name].active = True
                mesh.uv_layers[origin_uv_name].active_render = True


def unset_barycentric_coord_uv(objs, backup_table):
    obj_list = retrieve_all(objs)

    for obj in obj_list:
        blender_obj = obj.blender_obj
        if blender_obj.type == 'MESH':
            mesh = blender_obj.data
            remove_uv = mesh.uv_layers.get('BarycentricCoord')
            if remove_uv is not None:
                mesh.uv_layers.remove(remove_uv)
            if blender_obj.name in backup_table:
                origin_uv_name = backup_table[blender_obj.name]
                mesh.uv_layers[origin_uv_name].active = True
                mesh.uv_layers[origin_uv_name].active_render = True


def triangulate_objs(objs):
    obj_list = retrieve_all(objs)

    for obj in obj_list:
        blender_obj = obj.blender_obj
        if blender_obj.type == 'MESH':
            # cleanup_mesh(blender_obj)

            me = blender_obj.data
            bm = bmesh.new()
            bm.from_mesh(me)

            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')

            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # Finish up, write the bmesh back to the mesh
            bm.to_mesh(me)
            bm.free()


def disable_output(output_type):
    assert output_type in ['uv', 'depth', 'segmentation', 'normal']
    type_table = {
        'uv': ['UV', 'use_pass_uv', 'uv'],
        'depth': ['Depth', 'use_pass_z', 'depth'],
        'segmentation': ['IndexOB', 'use_pass_object_index', 'segmap'],
        'normal': ['Normal', 'use_pass_normal', 'normals'],
    }
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, 'CompositorNodeRLayers')
    output = render_layer_node.outputs[type_table[output_type][0]]

    for link in links:
        if link.from_node == output:
            to_node = link.to_node
            links.remove(link)
            nodes.remove(to_node)
            break

    setattr(bpy.context.view_layer, type_table[output_type][1], False)

    remove_output_entry_by_key(type_table[output_type][2])


def remove_output_entry_by_key(key):
    if GlobalStorage.is_in_storage("output"):
        output_list = GlobalStorage.get("output")
        new_output = []
        for _output in output_list:
            if _output["key"] != key:
                new_output.append(_output)
        GlobalStorage.set("output", new_output)


def set_category_id(obj_list):
    ct = 0
    for obj in obj_list:
        blender_obj = obj.blender_obj
        is_root = (blender_obj.parent is None) or (blender_obj.parent.type == 'EMPTY')
        if is_root:
            ct += 1
            obj.set_cp('category_id', ct)
            if blender_obj.type == 'EMPTY':
                continue
            if hasattr(obj, 'get_children'):
                childs = obj.get_children(return_all_offspring=True)
                for child in childs:
                    # if not child.has_cp('category_id'):
                    child.set_cp('category_id', ct)