import os
import cv2
import numpy as np
import open3d as o3d

class TMPPDataset:
    def __init__(self, root_env_dir, image_size=(1024,1024),depth_scale=1000.0) -> None:
        self.root_env_dir = root_env_dir
        self.depth_scale = depth_scale
        self.image_size = image_size
    
    def __call__(self, seq_name):
        self.seq_name = seq_name
        self.data_root_dir = self.get_data_root_dir()

        self.rgb_dir = os.path.join(self.data_root_dir, 'color')
        self.depth_dir = os.path.join(self.data_root_dir, 'depth')
        self.segmentation_dir = os.path.join(self.data_root_dir, 'mask')
        self.object_dict_dir = os.path.join(self.data_root_dir, 'object')
        self.point_cloud_path = os.path.join(self.data_root_dir, 'point_cloud.ply')
        self.extrinsics_dir = os.path.join(self.data_root_dir, 'pose')
        self.intrinsic_dir = os.path.join(self.data_root_dir, 'intrinsic')

        return self
        

    def get_data_root_dir(self):
        data_root_dir = os.path.join(self.root_env_dir, self.seq_name.split('_')[0], self.seq_name.split('_')[1], self.seq_name.split('_')[2], 'frames')
        return data_root_dir


    def get_frame_list(self, stride):
        image_list = os.listdir(self.rgb_dir)
        image_list = sorted(os.listdir(self.rgb_dir), key=lambda x: int(x.split('.')[0]))
        frame_ids = [x.split('.')[0] for x in image_list][::stride]

        return frame_ids
    

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.intrinsic_dir}/intrinsic_depth.txt'
        intrinsics = np.loadtxt(intrinsic_path)

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(self.image_size[0], self.image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        pose_path = os.path.join(self.extrinsics_dir, str(frame_id) + '.txt')
        pose = np.loadtxt(pose_path)
        return pose
    

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, str(frame_id) + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth


    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        if align_with_depth:
            segmentation = cv2.resize(segmentation, self.image_size, interpolation=cv2.INTER_NEAREST)
        return segmentation


    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        return rgb_path, segmentation_path


    def get_scene_points(self):
        mesh = o3d.io.read_point_cloud(self.point_cloud_path)
        vertices = np.asarray(mesh.points)
        return vertices
    
    def get_scene_list(self):
        scene_list=[]
        for scene_id in os.listdir(self.root_env_dir):
            if not os.path.isdir(os.path.join(self.root_env_dir, scene_id)):
                continue
            for house_id in os.listdir(os.path.join(self.root_env_dir, scene_id)):
                for room_id in os.listdir(os.path.join(self.root_env_dir, scene_id, house_id)):
                    scene_list.append((scene_id, house_id, room_id))
        return sorted(scene_list)
    
    def get_if_result_dir(self, cfg, scene_id):
        if_root_env_dir = cfg["data_path"]["if_root_env_dir"]
        result_dir = os.path.join(if_root_env_dir, *scene_id, "tasmap_pp_results")
        return result_dir
    
    def get_benchmark_file_path(self, cfg, scene_id):
        benchmark_root_dir = cfg["evaluation"]["benchmark_root_dir"]
        npy_file_name = "_".join(map(str, scene_id))+".npy"
        benchmark_file_path = os.path.join(benchmark_root_dir, npy_file_name)
        return benchmark_file_path
    
    def get_result_file_path(self, cfg, scene_id):
        result_dir = self.get_if_result_dir(cfg, scene_id)
        result_file_path = os.path.join(result_dir, "tasmap_pp.npy")
        return result_file_path