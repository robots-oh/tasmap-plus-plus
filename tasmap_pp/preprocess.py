import os
import shutil
from tqdm import tqdm

import cv2
import torch
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from collections import Counter
from utils.mask_predict import MaskPredictor

class Preprocess():
    """
    3d preprocess and mask prediction
    """
    def __init__(self, preprocessed_root_path, save_root_path, camera_config, pcd_vis_flag, seg_GT_flag=False):
        self.camera_height = camera_config['camera_height']
        self.camera_width = camera_config['camera_width']
        self.focal_length = camera_config['focal_length']
        self.horiz_aperture = camera_config['horiz_aperture']

        self.seg_GT_flag = seg_GT_flag

        pcd = self.__create_downsampled_point_cloud(preprocessed_root_path,
                                    image_size=(self.camera_width, self.camera_height),
                                    stride=1,
                                    voxel_down_sample_size=0.005,
                                    image_ds_factor=8,
                                    depth_scale=1000)
        
        output_ply_path = os.path.join(save_root_path, f'point_cloud.ply')
        o3d.t.io.write_point_cloud(output_ply_path, pcd)
        
        if pcd_vis_flag:
            # o3d.visualization.draw_geometries([pcd], window_name="Downsampled Point Cloud")
            app = o3d.visualization.gui.Application.instance
            app.initialize()

            window = o3d.visualization.O3DVisualizer("My PointCloud", 1024, 768)
            window.add_geometry("point cloud", pcd)

            app.add_window(window)
            app.run()

        mask_predictor = MaskPredictor()
        mask_predictor(data_root_path=preprocessed_root_path,
                    debug_image_save_flag=True)
        
    def __get_intrinsics(self, image_size, intrinsic_path):
        intrinsics = np.loadtxt(intrinsic_path)
        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(image_size[0], image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters


    def __get_depth(self, depth_path, depth_scale):
        depth = cv2.imread(depth_path, -1)
        depth = depth / depth_scale
        depth = depth.astype(np.float32)
        return depth


    def __scale_intrinsics(self, intrinsic:o3d.camera.PinholeCameraIntrinsic, factor:int):
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()
        return o3d.camera.PinholeCameraIntrinsic(
            width  = intrinsic.width  // factor,
            height = intrinsic.height // factor,
            fx     = fx / factor,
            fy     = fy / factor,
            cx     = cx / factor,
            cy     = cy / factor,
        )

    def __downsample_color_and_depth(self, color_np: np.ndarray, depth_np: np.ndarray, seg_np: np.ndarray, factor: int):
        color_ds = color_np[::factor, ::factor].copy()      # (H/f, W/f, 3)
        depth_ds = depth_np[::factor, ::factor].copy()      # (H/f, W/f)
        seg_ds = seg_np[::factor, ::factor].copy()   
        return color_ds, depth_ds, seg_ds


    def __voxel_downsample_with_dominant_seg_id(self, pcd_tensor, voxel_size):
        points = np.asarray(pcd_tensor.point["positions"].numpy())
        seg_ids = np.asarray(pcd_tensor.point["segment_GT_id"].numpy()).flatten()
        colors = np.asarray(pcd_tensor.point["colors"].numpy())

        min_bound = points.min(axis=0)
        voxel_coords = np.floor((points - min_bound) / voxel_size).astype(int)

        voxel_dict = {}
        for idx, coord in tqdm(enumerate(map(tuple, voxel_coords))):
            if coord not in voxel_dict:
                voxel_dict[coord] = {"points": [], "seg_ids": [], "colors": []}
            voxel_dict[coord]["points"].append(points[idx])
            voxel_dict[coord]["seg_ids"].append(seg_ids[idx])
            voxel_dict[coord]["colors"].append(colors[idx])

        downsampled_points = []
        downsampled_seg_ids = []
        downsampled_colors = []

        for coord, voxel in tqdm(voxel_dict.items()):
            coord_np = np.array(coord)
            voxel_center = (coord_np + 0.5) * voxel_size + min_bound
            avg_color = np.mean(voxel["colors"], axis=0)
            seg_id_mode = Counter(voxel["seg_ids"]).most_common(1)[0][0]

            downsampled_points.append(voxel_center)
            downsampled_colors.append(avg_color)
            downsampled_seg_ids.append([seg_id_mode])

        new_pcd = o3d.t.geometry.PointCloud()
        new_pcd.point["positions"] = o3d.core.Tensor(downsampled_points, dtype=o3d.core.Dtype.Float32)
        new_pcd.point["colors"] = o3d.core.Tensor(downsampled_colors, dtype=o3d.core.Dtype.Float32)
        new_pcd.point["segment_GT_id"] = o3d.core.Tensor(downsampled_seg_ids, dtype=o3d.core.Dtype.Int32)

        return new_pcd

    def __backproject(self, color, depth, seg, intrinisc_cam_parameters, extrinsics, ds_factor=1):
        color, depth, seg = self.__downsample_color_and_depth(color, depth, seg, ds_factor)
        intrinsic_ds = self.__scale_intrinsics(intrinisc_cam_parameters, ds_factor)

        depth_limit = 20
        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(color)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_limit,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic_ds
        )
        pcd.transform(extrinsics)

        valid_mask = (np.asarray(depth) > 0) & (np.asarray(depth) < depth_limit)
        valid_indices = np.where(valid_mask.flatten())[0]
        pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(pcd)

        if self.seg_GT_flag:
            seg_flat = seg.flatten()[valid_indices]
            seg_reshaped = seg_flat.reshape((-1, 1))
            pcd_tensor.point["segment_GT_id"] = o3d.core.Tensor(seg_reshaped)

        # positions_shape = pcd_tensor.point["positions"].shape[0]
        # segment_id_shape = seg_reshaped.shape[0]
        # assert positions_shape == segment_id_shape, (
        #     f"Shape mismatch: positions={positions_shape}, segment_GT_id={segment_id_shape}"
        # )
        return pcd_tensor


    def __create_downsampled_point_cloud(self,
                                         base_dir,
                                         image_size,
                                         stride,
                                         voxel_down_sample_size,
                                         image_ds_factor,
                                         depth_scale):
        depth_dir = os.path.join(base_dir, 'depth')
        if self.seg_GT_flag:
            seg_dir = os.path.join(base_dir, 'seg')
        color_dir = os.path.join(base_dir, 'color')
        pose_dir = os.path.join(base_dir, 'pose')
        intr_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')

        image_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('.')[0]))

        frame_ids = [x.split('.')[0] for x in image_list][::stride]
        intrinisc_cam_parameters = self.__get_intrinsics(image_size, intr_path)

        # full_pcd = o3d.geometry.PointCloud()
        full_pcd = None
        with tqdm(total=len(frame_ids)) as pbar:
            for i, fid in enumerate(frame_ids):
                pbar.set_description(f"Pointcloud (Frame {fid})")
                pbar.update(1)

                depth_path = os.path.join(depth_dir, f"{fid}.png")
                depth = self.__get_depth(depth_path, depth_scale)

                if self.seg_GT_flag:
                    seg_path = os.path.join(seg_dir, f"{fid}.png")
                    seg = cv2.imread(seg_path, -1)
                else:
                    seg = np.zeros_like(depth, dtype=np.int32)

                color = imageio.v2.imread(os.path.join(color_dir, f"{fid}.jpg"))
                # Resize color image to match depth image shape
                if color.shape[:2] != depth.shape:
                    color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

                extrinsics = np.loadtxt(os.path.join(pose_dir, f"{fid}.txt"))

                pcd = self.__backproject(color, depth, seg, intrinisc_cam_parameters, extrinsics, ds_factor=image_ds_factor)
                # full_pcd += pcd
                if full_pcd is None:
                    full_pcd = pcd.clone()
                else:
                    try:
                        full_pcd = full_pcd + pcd 
                    except:
                        continue


        if self.seg_GT_flag:
            full_pcd = self.__voxel_downsample_with_dominant_seg_id(full_pcd, voxel_down_sample_size)
        else:
            full_pcd = full_pcd.voxel_down_sample(voxel_size=voxel_down_sample_size)
        return full_pcd







OMNI_SENSOR_HEIGHT = 1024
OMNI_SENSOR_WIDTH = 1024
OMNI_FOCAL_LENGTH = 17.0
OMNI_HORIZ_APERTURE = 20.954999923706055

if __name__=="__main__":

    data_root_dir = '/workspace/data/scenes/disordered/A/Bathroom-18516/frames'
    camera_config = {
        'camera_height':OMNI_SENSOR_HEIGHT,
        'camera_width':OMNI_SENSOR_WIDTH,
        'focal_length':OMNI_FOCAL_LENGTH,
        'horiz_aperture':OMNI_HORIZ_APERTURE
    }

    Preprocess(preprocessed_root_path=data_root_dir,
                save_root_path=data_root_dir,
                camera_config=camera_config,
                pcd_vis_flag=True,
                seg_GT_flag=False)

