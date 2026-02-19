"""
Omnigibson simulation utility functions

"""
import os
import sys
import json
import numpy as np

from .macros import tp
from .math_utils import quaternion_rotation_matrix
from omnigibson.objects import USDObject, LightObject
from omnigibson.utils.python_utils import create_object_from_init_info


def water_obj(label_code, USDObject, scale):
    """
    sink의 "ON" state를 위한 water obj load 함수
    """
    asset_path = os.path.join(tp.CUS_USD_PATH, 'sink_water_cojahh.usd')
    label_code = f'water_{label_code}'
    obj = USDObject(
        name=label_code,
        scale = scale,
        usd_path=asset_path,
        visual_only=True,
        fixed_base=True
        )
    
    return obj

def flame_obj(label_code, LightObject, radius):
    """
    stove의 "ON" state를 위한 light obj load 함수
    디스크 형태의 빛 생성
    """
    obj = LightObject(
        name = label_code,
        light_type = 'Disk',
        radius = radius,
        intensity = 2000,
        fixed_base = True,
    )
    
    return obj

def light_obj(label_code, LightObject):
    """
    light stand, lamp 등의 물체의 "ON" state를 위한 light obj load 함수
    구체의 빛 생성
    """
    label_code = f'light_{label_code}'
    obj = LightObject(
        name = label_code,
        light_type = 'Sphere',
        radius = 0.1,
        fixed_base = True,
    )
    return obj

def dataset_obj(label_code, scale=[1,1,1]):
    """
    og_dataset object
    """
    category = "_".join(label_code.split("_")[:-2])
    model = label_code.split("_")[-2]
    init_info = {
        "class_module": "omnigibson.objects.dataset_object",
        "class_name": "DatasetObject",
        "args": {
            "name": label_code,
            "prim_path": f"/World/{label_code}",
            "category": category,
            "model": model,
            "scale": scale,
            "fixed_base": True,
            "visual_only": True,
            "in_rooms": [],
            "bddl_object_scope": None
        }
    }
    return init_info

def custom_obj(label_code, USDObject, scale=[1,1,1]):
    """
    custom object
    """
    category = "_".join(label_code.split("_")[:-2])
    model = label_code.split("_")[-2]
    asset_path = os.path.join(tp.CUS_USD_PATH, f'{category}_{model}.usd')
    obj = USDObject(
        name=label_code,
        scale = scale,
        usd_path=asset_path,
        visual_only=True,
        fixed_base=True
        )
    return obj

def base_obj(label_code, USDObject, scale=[1,1,1]):
    """
    floors and walls
    """
    category = "_".join(label_code.split("_")[:-2])
    model = label_code.split("_")[-2]
    asset_path = os.path.join(tp.CUS_USD_PATH, f'{category}_{model}.usd')
    obj = USDObject(
        name=label_code,
        scale = scale,
        usd_path=asset_path,
        visual_only=True,
        fixed_base=True
        )
    return obj

def collision_obj(label_code, USDObject, scale=[1,1,1]):
    """
    additional collision obj for floors
    """
    asset_path = os.path.join(tp.CUS_USD_PATH, 'cus_floors_collision.usd')
    obj = USDObject(
        name=f'NAN_collision_{label_code}',
        scale = scale,
        usd_path=asset_path,
        visual_only=True,
        fixed_base=True
        )
    return obj

def load_objects_into_scene(og, objects_in_scene):
    """
    load objects into scene
    return transparent obj list, open obj list
    """
    with open(os.path.join(tp.FW_TEXTURE_PATH, 'texture_scale.json'), 'r') as jf:
        texture_scale = json.load(jf)

    with open(os.path.join(tp.OBJ_DATA_PATH, 'state_preset.json'), 'r') as jf:
        state_preset = json.load(jf)

    for label_code, item in objects_in_scene.items():
        if item['class_name'] == 'BaseObject':
            # if 'cus_floors' in label_code:
            obj = collision_obj(label_code, USDObject, item['scale'])
            og.sim.import_object(obj)
            obj.set_position_orientation(
                position=item['pos'],
                orientation=item['ori'],
            )    
            
            obj = base_obj(label_code, USDObject, item['scale'])
            og.sim.import_object(obj)
            obj.set_position_orientation(
                position=item['pos'],
                orientation=item['ori'],
            )
            if 'floor' in label_code and item['texture']:
                texture_path = os.path.join(tp.FW_TEXTURE_PATH, item['texture']+texture_scale[item['texture']][0])
                s_scale = texture_scale[item['texture']][1]*item['scale'][0]
                t_scale = texture_scale[item['texture']][2]*item['scale'][1]
                og.sim.stage.GetAttributeAtPath(f"/World/{label_code}/cube/Looks/OmniSurface/Shader.inputs:diffuse_reflection_color_image").Set(texture_path)
                og.sim.stage.GetAttributeAtPath(f"/World/{label_code}/cube/Looks/OmniSurface/Shader.inputs:uvw_s_scale").Set(s_scale)
                og.sim.stage.GetAttributeAtPath(f"/World/{label_code}/cube/Looks/OmniSurface/Shader.inputs:uvw_t_scale").Set(t_scale)
                og.sim.stage.GetAttributeAtPath(f"/World/{label_code}/cube/Looks/OmniSurface/Shader.inputs:specular_reflection_ior").Set(1.0)

            elif 'wall' in label_code and item['texture']:
                color = tp.SC_WALL_COLOR[item['texture']]
                og.sim.stage.GetAttributeAtPath(f"/World/{label_code}/cube/Looks/material/Shader.inputs:diffuse_color_constant").Set(color)
                
    transparent_list = []
    open_list = []
    for label_code, item in objects_in_scene.items():
        if item['class_name'] == 'BaseObject' : continue
        catergory = "_".join(label_code.split("_")[:-2])

        if item['class_name'] == 'DatasetObject':
            init_info = dataset_obj(label_code, item['scale'])
            obj = create_object_from_init_info(init_info)

        else:
            if 'liquid' in label_code:
                transparent_list.append(label_code)
            obj = custom_obj(label_code, USDObject, item['scale'])

        try:
            og.sim.import_object(obj)
        except Exception as e:
            print(f"Error importing object {label_code}: {e}")
            continue
        obj.set_position_orientation(
            position=item['pos'],
            orientation=item['ori'],
        )

        og.sim.stage.GetAttributeAtPath(f"{obj.root_link._prim_path}.physxRigidBody:disableGravity").Set(True)

        if 'light' in list(item['joints'].keys()):
            pos = item['joints']['light']
            obj = light_obj(label_code, LightObject)
            og.sim.import_object(obj)
            obj.set_position(
                position=pos
                )
        
        if 'water' in list(item['joints'].keys()):
            catergory = "_".join(label_code.split("_")[:-2])
            model = label_code.split("_")[-2]

            water_offset = np.array(state_preset[catergory][model]['offset'])
            water_offset *= -1 #[2]
            water_scale = np.array(state_preset[catergory][model]['scale'])

            water_scale = water_scale*np.array(item['scale'])
            water_offset = water_offset*np.array(item['scale'])

            rot_mat = quaternion_rotation_matrix(item['ori'])
            water_offset = np.matmul(rot_mat, water_offset)
            
            water_pos = item['pos']+water_offset
            
            obj = water_obj(label_code, USDObject, water_scale)
            og.sim.import_object(obj)
            obj.set_position(
                position=water_pos
                )
        
        if 'flame' in list(item['joints'].keys()):
            model = label_code.split("_")[-2]
            for light, light_item in state_preset['stove'][model].items():
                light_type = light.split("_")[0]
                light_offset = np.array(light_item['offset'])*np.array(item['scale'])

                rot_mat = quaternion_rotation_matrix(item['ori'])
                light_offset = np.matmul(rot_mat,light_offset)

                light_pos = np.array(item['pos'])+light_offset
                light_radius = light_item['radius']

                obj = flame_obj(f'flame_{light}_{label_code}', LightObject, light_radius)
                og.sim.import_object(obj)
                obj.set_position_orientation(
                    position=light_pos,
                    orientation=[0,-1,0,0]
                    )
                scale = tuple(item['scale'])
                og.sim.stage.GetAttributeAtPath(f"/World/flame_{light}_{label_code}/base_link/light.xformOp:scale").Set(scale)

                if light_type == 'r':
                    og.sim.stage.GetAttributeAtPath(f"/World/flame_{light}_{label_code}/base_link/light.inputs:color").Set((1.0, 0.0, 0.0))
                    og.sim.stage.GetAttributeAtPath(f"/World/flame_{light}_{label_code}/base_link/light.inputs:exposure").Set(10)
                else:
                    og.sim.stage.GetAttributeAtPath(f"/World/flame_{light}_{label_code}/base_link/light.inputs:color").Set((1.0, 0.83, 0.0))
                    og.sim.stage.GetAttributeAtPath(f"/World/flame_{light}_{label_code}/base_link/light.inputs:exposure").Set(5)

        if item['state'] == 'open':
            open_list.append(label_code)
            
    return transparent_list, open_list
