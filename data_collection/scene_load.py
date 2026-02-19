import sys
import os

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
import random

from sim_scripts.sim_utils import load_objects_into_scene

import json
import numpy as np

gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True

def get_dirs(path):
    return sorted([entry.name for entry in os.scandir(path) if entry.is_dir()])

def choose_from_options(options, name, random_selection=False):
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not random_selection:
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            k = min(max(int(s), 1), len(options)) - 1
        except ValueError:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    else:
        k = random.choice(range(len(options)))

    return list(options)[k]

class SceneLoader():
    def __init__(self, scene_dir=None, root_dir="/workspace/data/scenes", is_house=False):

        if scene_dir is None:
            scene_id = choose_from_options(get_dirs(root_dir), "Scenario", random_selection=False)
            house_id = choose_from_options(get_dirs(os.path.join(root_dir, scene_id)), "House", random_selection=False)
            if is_house:
                scene_lists = [(os.path.join(root_dir, scene_id, house_id, room_id), scene_id, house_id, room_id) for room_id in get_dirs(os.path.join(root_dir, scene_id, house_id))]
                self.msg = f"{scene_id} / {house_id}"
            else:
                room_id = choose_from_options(get_dirs(os.path.join(root_dir, scene_id, house_id)), "Room", random_selection=False)
                scene_dir = os.path.join(root_dir, scene_id, house_id, room_id)
                scene_lists = [(scene_dir, scene_id, house_id, room_id)]
                self.msg = f"{scene_id} / {house_id} / {room_id}"

        else:
            path, room_id = os.path.split(scene_dir)
            path, house_id = os.path.split(path)
            _, scene_id = os.path.split(path)
            scene_lists = [(scene_dir, scene_id, house_id, room_id)]
            self.msg = f"{scene_id} / {house_id} / {room_id}"

        self.scene_dir = scene_dir
        self.viewer_height = 1024
        self.viewer_width = 1024

        cfg = {
            "scene": {
                "type": "Scene",
            },
            "render": {
                "viewer_height": self.viewer_height,
                "viewer_width": self.viewer_width,
            },
        }
        self.env = og.Environment(configs=cfg)
        og.sim.stop()


        self.objects_in_scene = {}
        for idx, (scene_dir, scene_id, house_id, room_id) in enumerate(scene_lists):
            scene_json_path = os.path.join(scene_dir, f'{scene_id}_{house_id}_{room_id}_scene.json')
            with open(scene_json_path, 'r') as jf:
                room_scene_dict = json.load(jf)

            if is_house:
                with open(f'/workspace/data/house_json/{house_id}.json', 'r') as jf:
                    house_info = json.load(jf)

                offset = house_info[room_id]['loc']
                if len(offset) == 2:
                    offset.append(0.0)
                offset[1] *= -1

                for keys, items in room_scene_dict.items():
                    items['pos'] = [a + b for a, b in zip(items['pos'], offset)]
                    items['origin'] = f'{keys}--{room_id}'
                    self.objects_in_scene[f'{keys}{idx}'] = items
            else:
                self.objects_in_scene = room_scene_dict

        self.transparent_list, self.open_list = load_objects_into_scene(og, self.objects_in_scene)
        self.env.scene._floor_plane.remove()
        
        self.stage = og.sim.stage
        
        cam_mover = og.sim.enable_viewer_camera_teleoperation()
        cam_mover.cam = og.sim.viewer_camera
        self.cam = cam_mover.cam
        self.cam.add_modality('rgb')
        self.cam.add_modality('seg_semantic')
        self.cam.add_modality('depth_linear')
        self.cam.add_modality('seg_instance')
        
        self.cam.set_position_orientation(
            position=np.array([0, 0, 7]),
            orientation=np.array([0, 0, 0, 1])
        )

        og.sim.scene._skybox._light_link.set_attribute('inputs:exposure', 2)
        og.sim.play()
        
        for obj in og.sim.scene.objects:
            for link_name in obj._links:
                if link_name.startswith("lights") or link_name.startswith("togglebutton"):
                    obj._links[link_name].set_attribute('visibility', 'invisible')
                    prim = obj._links[link_name].prim
                    for child in prim.GetChildren():
                        visibility_attr = child.GetAttribute('visibility')
                        if not visibility_attr:
                            visibility_attr = child.CreateAttribute('visibility', lazy.pxr.Sdf.ValueTypeNames.Token)
                        visibility_attr.Set('invisible')

            
            for joint_name in obj._joints:
                if joint_name.startswith("j_"):
                    if obj._joints[joint_name].joint_type == 'PrismaticJoint':
                        og.sim.stage.GetAttributeAtPath(f"/World/{obj.name}/base_link/{joint_name}.drive:linear:physics:damping").Set(100)
                    elif obj._joints[joint_name].joint_type == 'RevoluteJoint':
                        og.sim.stage.GetAttributeAtPath(f"/World/{obj.name}/base_link/{joint_name}.drive:angular:physics:damping").Set(100)

        og.sim.play()
        for label_code in self.transparent_list:
            mesh = self.stage.GetPrimAtPath(f"/World/{label_code}/liquid")
            primvars_api = lazy.pxr.UsdGeom.PrimvarsAPI(mesh)
            primvars_api.CreatePrimvar("doNotCastShadows", lazy.pxr.Sdf.ValueTypeNames.Bool).Set(True)

        for label_code in self.open_list:
            for j_link, value in self.objects_in_scene[label_code]['joints'].items():
                obj = og.sim.scene.object_registry("name", label_code)
                obj._joints[j_link].set_pos(value)

        
        KeyboardEventHandler.initialize()
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.ESCAPE,
            callback_fn=lambda: self.env.close(),
        )

        for _ in range(10):
            self.env.step(action=[0,0])

    
    def simulate(self):
        print(f"\n### [CURRENT SCENE] : {self.msg} ###")
        while True:
            self.env.step(action=[0,0]) 

if __name__ == "__main__":
    
    # scene_dir = '/workspace/data/scenes/disordered/A/Bathroom-18516'
    # scene_loader = SceneLoader(scene_dir=scene_dir)
    # scene_loader.simulate()

    scene_loader = SceneLoader()
    scene_loader.simulate()

    