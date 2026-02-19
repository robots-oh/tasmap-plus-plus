import os
import sys

import json
import numpy as np
import copy
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import shutil
from PIL import Image


# dir_path = [os.path.dirname(os.path.realpath(__file__))]
# sys.path.append(os.path.split(dir_path[0])[0])
# from sim_scripts.sim_utils import BasePath
# BASE_PATH = BasePath(True)
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T

from sim_scripts.macros import tp
from scene_load import SceneLoader

import cv2
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

gm.DEFAULT_VIEWER_WIDTH = 1024
gm.DEFAULT_VIEWER_HEIGHT = 1024

WAYPOINT_MASK=90 #70
MIN_WAYPOINT_MASK=60
MAX_WAYPOINT=25 #25
MIN_WAYPOINT=7

ANGLE_INTERVAL = 30
MIN_DEPTH = 0.4
TH_PROPER_RATIO = 0.4

OBJ_DATA_PATH = "/workspace/data/obj_data"

import torch


def save_rgb(save_root_dir, count, cam):
    frame_num = "{:08}".format(count)
    cam_obs = cam.get_obs()

    # rgb_color_dir = os.path.join(save_root_dir, 'temp_color')
    # os.makedirs(rgb_color_dir, exist_ok=True)
    # rgb_image_path = os.path.join(rgb_color_dir, f'{frame_num}.png')
    # cv2.imwrite(rgb_image_path, cv2.cvtColor(np.array(cam_obs[0]["rgb"], dtype=np.uint8), cv2.COLOR_BGR2RGB))

    # rgb_color_dir = os.path.join(save_root_dir, 'color')
    # os.makedirs(rgb_color_dir, exist_ok=True)
    # output_image_path = os.path.join(rgb_color_dir, f'{frame_num}.jpg')
    # shutil.copy(rgb_image_path, output_image_path)

    rgb_color_dir = os.path.join(save_root_dir, 'color')
    os.makedirs(rgb_color_dir, exist_ok=True)
    rgb_image_path = os.path.join(rgb_color_dir, f'{frame_num}.jpg')
    cv2.imwrite(rgb_image_path, cv2.cvtColor(np.array(cam_obs[0]["rgb"], dtype=np.uint8), cv2.COLOR_BGR2RGB))


def save_tensor_to_txt(tensor, path):
    with open(path, "w") as f:
        for row in tensor:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")


def quaternion_rotation_matrix(Q, torch_=False):
    q0, q1, q2, q3 = Q[3], Q[0], Q[1], Q[2]
    
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    if torch_:
        return torch.tensor([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], device=tp.DEVICE, dtype=torch.float32)
    else:
        return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)

def extrinsic_matrix_torch(c_abs_ori, c_abs_pose):
    c_abs_pose = torch.as_tensor(c_abs_pose, device=tp.DEVICE, dtype=torch.float32).view(3, 1)
    rotation = quaternion_rotation_matrix(c_abs_ori, torch_=True)

    x_vector = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0], device=tp.DEVICE).T)
    y_vector = torch.matmul(rotation, torch.tensor([0.0, -1.0, 0.0], device=tp.DEVICE).T)
    z_vector = torch.matmul(rotation, torch.tensor([0.0, 0.0, -1.0], device=tp.DEVICE).T)
    
    rotation_matrix = torch.stack((x_vector, y_vector, z_vector))  # R

    # Translation vector
    translation_vector = -torch.matmul(rotation_matrix, c_abs_pose.view(3, 1))

    # Construct extrinsic matrix RT (4x4)
    RT = torch.eye(4, device=tp.DEVICE)
    RT[:3, :3] = rotation_matrix
    RT[:3, 3] = translation_vector.squeeze()

    # Compute RT inverse (4x4)
    RT_inv = torch.eye(4, device=tp.DEVICE)
    RT_inv[:3, :3] = rotation_matrix.T
    RT_inv[:3, 3] = torch.matmul(rotation_matrix.T, -translation_vector).squeeze()

    return RT, RT_inv


    



class SceneCreator(SceneLoader):
    def __init__(self, scene_dir, save_seg_info=False, waypoint_mask_circle_size=WAYPOINT_MASK, cover_size=WAYPOINT_MASK):
        super().__init__(scene_dir=scene_dir)
        self.save_seg_info = save_seg_info

        self.save_root_dir = os.path.join(self.scene_dir, 'frames')
        if os.path.isdir(self.save_root_dir):
            shutil.rmtree(self.save_root_dir)
        os.makedirs(self.save_root_dir, exist_ok=True)

        # Waypoint Parameter
        self.__depth_far_th=1.5
        self.__depth_close_th=0.5
        self.save_count=0
        self.waypoint=0
        self.step=0
        self.waypoint_mask_circle_size=waypoint_mask_circle_size
        self.cover_size=cover_size

        self.focal_length = 17
        self.horiz_aperture = 20.955
        self.vert_aperture = 15.2908


    def capture_scene(self):
        top_down_image = self.__save_top_down_image()
        self.__save_waypoint(top_down_image, waypoint_mask_circle=self.waypoint_mask_circle_size, waypoint_mask_cover=self.cover_size)
        self.__move_frame()
        self.__save_intrinsic()
        if self.save_seg_info:
            self.__save_total_seg_GT()


    def __save_top_down_image(self):
        self.cam.set_position_orientation(
            position=np.array([0, 0, 15]),
            orientation=np.array([0, 0, 0, 1])
        )

        og.sim.play()
        for _ in range(25):
            self.env.step(action=[0,0])  
        og.sim.pause()

        # Save RGB
        cam_obs = self.cam.get_obs()
        rgb = np.array(cam_obs[0]["rgb"], dtype=np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_root_dir, 'topdown_rgb.png'), rgb)

        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.projection").Set('orthographic')
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.horizontalAperture").Set(120)
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.focalLength").Set(50)
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.verticalAperture").Set(120)

        og.sim.play()
        for _ in range(15):
            self.env.step(action=[0,0])
        og.sim.pause()

        # Save Segmentation
        cam_obs = self.cam.get_obs()
        seg = np.array(cam_obs[0]["seg_instance"], dtype=np.uint8)
        top_down_image = np.zeros_like(seg)
        
        seg_id_label_GT = {}
        target_ids = []
        for id, label_code in cam_obs[1]["seg_instance"].items():
            id = int(id)
            seg_id_label_GT[label_code] = id
            if 'cus_floors' in label_code or 'carpet' in label_code:
                target_ids.append(id)

        for target_id in target_ids:
            temp = np.where(seg == target_id , 1, 0)
            top_down_image = top_down_image|temp

        top_down_image = np.array(top_down_image, dtype=np.uint8)*255

        # cv2.imwrite(os.path.join(self.save_root_dir, 'topdown.png'), top_down_image)

        return top_down_image

    def __save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def __save_intrinsic(self):

        # Intrinsic
        intrinsic_dir = os.path.join(self.save_root_dir, 'intrinsic')
        os.makedirs(intrinsic_dir, exist_ok=True)
        intrinsic_parameters = self.get_intrinsic_parameters(realsense=False)
        fx, fy, cx, cy = intrinsic_parameters
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        mat = np.eye(4)
        os.makedirs(intrinsic_dir, exist_ok=True)

        self.__save_mat_to_file(K, os.path.join(intrinsic_dir, 'intrinsic_color.txt'))
        self.__save_mat_to_file(mat, os.path.join(intrinsic_dir, 'extrinsic_color.txt'))
        self.__save_mat_to_file(K, os.path.join(intrinsic_dir, 'intrinsic_depth.txt'))
        self.__save_mat_to_file(mat, os.path.join(intrinsic_dir, 'extrinsic_depth.txt'))

    def __save_total_seg_GT(self):
        # Segmentation GT merge
        seg_info_dir = os.path.join(self.save_root_dir, 'seg_info')
        seg_total_data = {}
        for frame_seg_info_file in os.listdir(seg_info_dir):
            with open(os.path.join(seg_info_dir, frame_seg_info_file)) as sf:
                seg_data = json.load(sf)

            for key, data in seg_data.items():
                if data['GT_label_id'] not in seg_total_data:
                    seg_total_data[data['GT_label_id']] = int(key)

        seg_data_sorted = dict(sorted(seg_total_data.items(), key=lambda item: int(item[1])))

        with open(os.path.join(self.save_root_dir, 'total_seg_info.json'), mode = 'w') as jf:
            json.dump(seg_data_sorted, jf, indent=4)


    
    def ori2deg(self, ori):
        # ori: XYZW quat
        # deg: x,y,z
        rad = T.quat2euler(ori)
        deg = T.rad2deg(rad)
        return deg
    
    def deg2ori(self, deg):
        # deg: x,y,z
        # ori: XYZW quat
        rad = T.deg2rad(deg)
        ori = T.euler2quat(rad)
        return ori


    def __look_left(self, angle_interval, left=True):
        cur_pos, cur_ori=self.cam.get_position_orientation()
        for ang in range(0,360,angle_interval):
            if left:
                z_rad_ang = np.deg2rad(ang)
            else:
                z_rad_ang = np.deg2rad(-ang)

            rot_rad_angle = np.array([0, 0, z_rad_ang])
            rot_ori=T.axisangle2quat(rot_rad_angle)
            new_ori = T.quat_multiply(rot_ori, cur_ori)

            self.cam.set_local_pose(position=cur_pos, orientation=new_ori)

            for _ in range(5):
                _, _, _, _ = self.env.step(action=np.array([0,0]))
            self.__capture_scene()


    def __grad_move(self, goal_pos, goal_ori, grad_num=5):
        cur_pos, cur_ori=self.cam.get_position_orientation()
        # rad_diff=T.get_orientation_diff_in_radian(cur_ori, goal_ori)

        # for i in range(grad_num-1):
        #     new_pos = cur_pos + ((goal_pos-cur_pos)*i/grad_num)
        #     new_ori = T.quat_slerp(cur_ori, goal_ori, i/grad_num, shortestpath=True)
        #     self.cam.set_local_pose(position=new_pos, orientation=new_ori)
        #     for _ in range(5):
        #         self.env.step(action=np.array([0,0]))
        #     self.__capture_scene()

        self.cam.set_local_pose(position=goal_pos, orientation=goal_ori)
        for _ in range(10):
            self.env.step(action=np.array([0,0]))

    def get_intrinsic_parameters(self, realsense=False):
        vert_aperture = self.viewer_height/self.viewer_width * self.horiz_aperture
        focal_x = self.viewer_height * self.focal_length / vert_aperture
        focal_y = self.viewer_width * self.focal_length / self.horiz_aperture
        center_x = self.viewer_height * 0.5
        center_y = self.viewer_width * 0.5
        
        return (focal_x, focal_y, center_x, center_y)


    def save_seg(self, save_root_dir, count, cam, c_abs_pose, c_abs_ori, waypoint_info, obj_label_map, objects_in_scene):

        frame_num = "{:08}".format(count)
        cam_obs = cam.get_obs()

        # Depth
        depth_dir = os.path.join(save_root_dir, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        depth_image_path = os.path.join(depth_dir, f'{frame_num}.png')
        # # cv2.imwrite(depth_image_path, np.array(cam_obs[0]["depth_linear"]*1000, dtype=np.uint16))
        # depth_image = Image.fromarray(np.array(cam_obs[0]["depth_linear"]*1000, dtype=np.uint16))
        # depth_image.save(depth_image_path)

        depth_map = cam_obs[0]["depth_linear"].astype(np.float16)
        depth_scaled = (depth_map * 1000).astype(np.uint16)

        depth_image = Image.fromarray(depth_scaled)
        depth_image.save(depth_image_path)



        # Pose
        pose_dir = os.path.join(save_root_dir, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
        RotT, RotT_inv = extrinsic_matrix_torch(c_abs_ori, c_abs_pose)
        pose_txt_path = os.path.join(pose_dir, f'{frame_num}.txt')
        save_tensor_to_txt(RotT_inv, pose_txt_path)

        if self.save_seg_info:

            # Segmentation
            seg_dir = os.path.join(save_root_dir, 'seg')
            os.makedirs(seg_dir, exist_ok=True)
            seg_color_path = os.path.join(seg_dir, f'{frame_num}.png')
            cv2.imwrite(seg_color_path, np.array(cam_obs[0]["seg_instance"], dtype=np.uint8))
            
            # Segmentation info
            seg_info_dir = os.path.join(save_root_dir, 'seg_info')
            os.makedirs(seg_info_dir, exist_ok=True)
            segmentation_list = []
            seg_info = {}
            for id, label_code in cam_obs[1]['seg_instance'].items():
                id = int(id)
                label_code = str(label_code)
                if label_code in ['background', 'unlabelled']:
                    label = label_code
                    # tasmap_task = None
                else:   
                    label = "_".join(label_code.split("_")[:-2])
                    label = obj_label_map[label]
                    # tasmap_task = objects_in_scene[label_code]['task']
                idx = segmentation_list.count(label)
                segmentation_list.append(label)
                label = f'{label}_{idx}'
                
                seg_info[id] = {
                    'GT_label_id':label_code,
                    # 'tasmap_task':tasmap_task,    
                    "GT_label":label,
                }

            with open(os.path.join(seg_info_dir, f'{frame_num}.json'), mode = 'w') as jf:
                json.dump(seg_info, jf, indent=4)


    def __capture_scene(self):
        if self.__is_proper_distance():
        
            with open(os.path.join(OBJ_DATA_PATH, 'obj_label_mapping.json'), 'r') as jf:
                obj_label_map = json.load(jf)

            for label_code in self.transparent_list:
                mesh = self.stage.GetPrimAtPath(f"/World/{label_code}/liquid")
                primvars_api = lazy.pxr.UsdGeom.PrimvarsAPI(mesh)
                primvars_api.CreatePrimvar("doNotCastShadows", lazy.pxr.Sdf.ValueTypeNames.Bool).Set(True)

            for _ in range(5):
                self.env.step(action=np.array([0,0]))
            og.sim.render()
            save_rgb(self.save_root_dir, self.save_count, self.cam)
            for label_code in self.transparent_list:
                mesh = self.stage.GetPrimAtPath(f"/World/{label_code}/liquid")
                primvars_api = lazy.pxr.UsdGeom.PrimvarsAPI(mesh)
                primvars_api.CreatePrimvar("doNotCastShadows", lazy.pxr.Sdf.ValueTypeNames.Bool).Set(False)

            for _ in range(5):
                self.env.step(action=np.array([0,0]))
            og.sim.render()
            cur_pos, cur_ori = self.cam.get_position_orientation()
            self.save_seg(self.save_root_dir, self.save_count, self.cam, cur_pos, cur_ori, [self.waypoint, self.step], obj_label_map, self.objects_in_scene)     
            
            self.save_count += 1


    def __look_around(self, pos_ori):

        goal_pos = pos_ori.copy() 
        goal_pos.append(0)

        self.step=0
        goal_pos[2] = 1.5
        goal_ori = self.deg2ori(np.array([55, 0, 0]))
        self.__grad_move(goal_pos, goal_ori, grad_num=1)
        self.__look_left(ANGLE_INTERVAL, left=True)     # rotate left

        self.step=1
        goal_pos[2] = 0.8
        goal_ori = self.deg2ori(np.array([90, 0, 0])) 
        self.__grad_move(goal_pos, goal_ori, grad_num=1)
        self.__look_left(ANGLE_INTERVAL, left=False)    # rotate right

        self.step=2
        goal_pos[2] = 1.1
        goal_ori = self.deg2ori(np.array([90, 0, 0]))   
        self.__grad_move(goal_pos, goal_ori, grad_num=1)
        self.__look_left(ANGLE_INTERVAL, left=True)     # rotate left


                

    def __is_proper_distance(self):
        cam_obs = self.cam.get_obs()
        depth_linear = cam_obs[0]["depth_linear"]
        total_pixels = depth_linear.size
        close_pixels = np.count_nonzero(depth_linear < MIN_DEPTH)
        proper_ratio = 1.0 - (close_pixels / total_pixels)
        is_proper =  proper_ratio > TH_PROPER_RATIO

        return is_proper
        

    def __save_waypoint(self, top_down_image_original, save_waypoint_image=False, waypoint_mask_circle = 90, waypoint_mask_cover = 90):
        top_down_image = copy.deepcopy(top_down_image_original)
        _, threshold_img = cv2.threshold(top_down_image, 0, 50, cv2.THRESH_BINARY_INV)
        floodfill_img = threshold_img.copy()
        
        h, w = threshold_img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(floodfill_img, mask, (0,0), 255)
        
        floodfill_img_inv = cv2.bitwise_not(floodfill_img)
        filled_topdown = threshold_img | floodfill_img_inv
        
        _, binary_img = cv2.threshold(filled_topdown, 175, 255, 0)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        max = 0
        topdown_region = 0
        for i in range(len(contours)):
            temp = np.zeros_like(top_down_image)
            cv2.drawContours(temp, contours, i, color=(255, 255, 255), thickness=cv2.FILLED)
            count = np.where(temp == 255, 1, 0)
            count = np.sum(count)
            if count > max : 
                max = count
                topdown_region = np.copy(temp)
        
        waypoint_mask_circle_size = waypoint_mask_circle
        cover_size = waypoint_mask_cover
        while True:
            height, width = top_down_image.shape[:2]
            waypoint_mask = np.full((height, width), 255).astype('float32')
            map_waypoint_candidate_list = []

            for x_point in range(0, width, cover_size//2):
                if width-x_point < cover_size//2:
                    continue
                for y_point in range(0, height, cover_size//2):
                    if height-y_point < cover_size//2:
                        continue

                    left=x_point
                    top=y_point
                    top_left = (left, top)

                    right = min(x_point+cover_size, width)
                    bottom = min(y_point+cover_size, height)
                    bottom_right = (right, bottom)

                    clip_area = top_down_image.copy()
                    clip_area[:top_left[1], :] = 0
                    clip_area[bottom_right[1]:, :] = 0
                    clip_area[:, :top_left[0]] = 0
                    clip_area[:, bottom_right[0]:] = 0

                    clip_transform = cv2.distanceTransform(clip_area, cv2.DIST_L2, 5)
                    clip_transform = np.where(waypoint_mask == 255, clip_transform, 0)

                    if np.count_nonzero(clip_transform>0) > ((right-left)*(bottom-top))*0.5:
                        max_value = np.max(clip_transform)
                        max_indices = np.argwhere(clip_transform == max_value)

                        min_distance = 999
                        max_point_x = -1
                        max_point_y = -1
                        for index in max_indices:
                            distance = abs(x_point-index[1]) + abs(y_point-index[0])
                            if distance < min_distance:
                                max_point_x = index[1]
                                max_point_y = index[0]
                        cv2.circle(waypoint_mask, (max_point_x, max_point_y), waypoint_mask_circle_size, (0, 0, 0), -1)
                        if topdown_region[max_point_y][max_point_x] == 255:
                            try:
                                left_dist = list(reversed(top_down_image[max_point_y][:max_point_x])).index(0)
                                right_dist = list(top_down_image[max_point_y][max_point_x:]).index(0)
                                top_dist = list(reversed((top_down_image[:,max_point_x])[:max_point_y])).index(0)
                                bottom_dist = list((top_down_image[:,max_point_x])[max_point_y:]).index(0)
                                dist = np.min(np.array([left_dist, right_dist, top_dist, bottom_dist]))
                                map_waypoint_candidate_list.append([dist, max_point_x, max_point_y])
                            except:
                                pass

            # if len(map_waypoint_candidate_list) > MAX_WAYPOINT:
            #     map_waypoint_candidate_list = sorted(map_waypoint_candidate_list, key=lambda x: x[0])
            #     map_waypoint_list = map_waypoint_candidate_list[-MAX_WAYPOINT:]
            #     map_waypoint_list = np.array(map_waypoint_list)[:, 1:]
            # else:
            #     map_waypoint_list = np.array(map_waypoint_candidate_list)[:, 1:]
            map_waypoint_list = np.array(map_waypoint_candidate_list)[:, 1:]

            if len(map_waypoint_list) < MIN_WAYPOINT:
                waypoint_mask_circle_size -= 10
                cover_size -= 10

            else:
                waypoint_mask_circle_size -= 10
                cover_size -= 10
                if waypoint_mask_circle_size < MIN_WAYPOINT_MASK:
                    break

                if len(map_waypoint_list)>MAX_WAYPOINT:
                    break

        actual_waypoint_list = [[(point_x-512)*(10/1024), (point_y-512)*(10/1024)*-1] for point_x, point_y in map_waypoint_list]

        if save_waypoint_image:
            color_top_down_image = cv2.cvtColor(top_down_image, cv2.COLOR_GRAY2RGB)
            for i in range(len(map_waypoint_list)):
                point_x = map_waypoint_list[i][0]
                point_y = map_waypoint_list[i][1]
                cv2.circle(color_top_down_image, (point_x, point_y), 3, (255, 0, 0), -1)
                cv2.putText(color_top_down_image,f'{i}',(point_x, point_y-3),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.imwrite(os.path.join(self.save_root_dir, 'waypoint.png'), color_top_down_image)

        np.save(os.path.join(self.save_root_dir, 'waypoint.npy'), np.array(actual_waypoint_list))

        
        return map_waypoint_list, actual_waypoint_list


    def __move_frame(self):
        og.sim.pause()
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.projection").Set('perspective')
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.focalLength").Set(self.focal_length)
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.horizontalAperture").Set(self.horiz_aperture)
        og.sim.stage.GetAttributeAtPath("/World/viewer_camera.verticalAperture").Set(self.vert_aperture)
        og.sim.play()

        try:
            actual_waypoint_list = np.load(os.path.join(self.save_root_dir, 'waypoint.npy')).tolist()
        except:
            self.__save_created_info()
            actual_waypoint_list = np.load(os.path.join(self.save_root_dir, 'waypoint.npy')).tolist()

        self.obj_count = len(self.objects_in_scene.keys()) + 2

        for idx, pos in enumerate(tqdm(actual_waypoint_list)):
            self.waypoint=idx
            self.__look_around(pos)
                




def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", default="/workspace/data/scenes/post-shower/A/Bathroom-18516", help="scene folder path")
    parser.add_argument("--save_seg_info", default=False)
    args = parser.parse_args(argv)

    scene_creator = SceneCreator(args.scene_dir, save_seg_info=args.save_seg_info)
    scene_creator.capture_scene()


if __name__ == "__main__":
    main(sys.argv[1:])
