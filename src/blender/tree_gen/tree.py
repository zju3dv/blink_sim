import blenderproc as bproc
import numpy as np
import bpy
from src.blender.tree_gen import parametric
from src.blender.tree_gen.parametric.tree_params import acer, apple, balsam_fir, bamboo, black_oak, black_tupelo, \
    cambridge_oak, douglas_fir, european_larch, fan_palm, lombardy_poplar, hill_cherry, palm, quaking_aspen, sassafras, \
    silver_birch, small_pine, sphere_tree, weeping_willow, weeping_willow_o


params = [acer.params,
          apple.params,
          balsam_fir.params,
          bamboo.params,
          black_oak.params,
          black_tupelo.params,
          cambridge_oak.params,
          douglas_fir.params,
          european_larch.params,
          fan_palm.params,
          hill_cherry.params,
          lombardy_poplar.params,
          palm.params,
          quaking_aspen.params,
          sassafras.params,
          silver_birch.params,
          small_pine.params,
          sphere_tree.params,
          weeping_willow.params,
          weeping_willow_o.params]


def create_tree(material_dir):
    tree_type = int(np.random.random() * len(params))
    param = params[tree_type]

    if 'leaf_blos_num' in param:
        param['leaf_blos_num'] = int((0.1 + np.random.random()) * param['leaf_blos_num'])

    if 'branches' in param:
        for j in range(len(param['branches'])):
            if param['branches'][j] > 0:
                branch_num = max(1, int((0.1+np.random.random()) * param['branches'][j]))
                param['branches'][j] = branch_num

    tree = parametric.gen.construct(params[tree_type], material_dir)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    tree.select_set(True)
    bpy.context.view_layer.objects.active = tree
    bpy.ops.object.select_all(action='DESELECT')
    tree.select_set(True)
    for child in tree.children:
        child.select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.join()

    tree = bproc.python.types.MeshObjectUtility.convert_to_meshes([tree])[0]

    return tree
