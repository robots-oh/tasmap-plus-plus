import os

import torch
import numpy as np
import open3d as o3d
import open3d.visualization as vis

from utils.post_process import post_process
from utils.construction import mask_graph_construction
from utils.iterative_clustering import iterative_clustering
from utils.dataset import TMPPDataset

class SceneSegmentation():
    def __init__(self, root_env_dir, seq_name, save_root_path, stride=1):
        self.mask_visible_threshold = 0.2
        self.contained_threshold = 0.8
        self.undersegment_filter_threshold = 0.1
        self.view_consensus_threshold = 0.9
        self.point_filter_threshold = 0.5

        self.datasets = TMPPDataset(root_env_dir=root_env_dir)
        self.dataset = self.datasets(seq_name)
        self.save_root_path = save_root_path
        self.scene_points = self.dataset.get_scene_points()
        self.frame_list = self.dataset.get_frame_list(stride=stride)


    def run(self):
        with torch.no_grad():
            nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(self.scene_points,
                                                                                                            self.frame_list,
                                                                                                            self.dataset,
                                                                                                            self.mask_visible_threshold,
                                                                                                            self.contained_threshold,
                                                                                                            self.undersegment_filter_threshold)

            object_list = iterative_clustering(nodes,
                                               observer_num_thresholds,
                                               self.view_consensus_threshold)

            post_process(self.dataset,
                         object_list,
                         mask_point_clouds,
                         self.scene_points,
                         point_frame_matrix,
                         self.frame_list,
                         self.point_filter_threshold,
                         self.save_root_path)


    def scene_visualize(self):
        np.random.seed(35)
        voxel_size = 0.01
        class_agnostic_mask_path = os.path.join(self.save_root_path, 'class_agnostic_mask.npz')
        mesh = o3d.io.read_triangle_mesh(self.dataset.point_cloud_path)
        
        scene_points = np.asarray(mesh.vertices)
        scene_points = scene_points - np.mean(scene_points, axis=0)

        scene_colors = np.asarray(mesh.vertex_colors)
        scene_colors = np.power(scene_colors, 1/2.2)  # Brighten colors
        scene_colors = np.clip(scene_colors, 0, 1)

        class_agnostic_mask = np.load(class_agnostic_mask_path)
        masks = class_agnostic_mask['pred_masks']
        num_instances = masks.shape[1]

        label_colors, labels, centers, instances_list = [], [], [], []
        instance_colors = np.zeros_like(scene_colors)

        for idx in range(num_instances):
            mask = masks[:, idx]
            point_ids = np.where(mask)[0]

            point_ids, points, colors, label_color, center = self.__vis_one_object(point_ids, scene_points)
            instance_colors[point_ids] = label_color
            label_colors.append(label_color)
            labels.append(str(idx))
            centers.append(center)

            # Add individual instance point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            instance = [{'name':f'obj_{idx}','group' : 'instances', 'geometry':pcd}]

            instances_list += instance

        # RGB full point cloud (optional)
        full_scene = o3d.geometry.PointCloud()
        full_scene.points = o3d.utility.Vector3dVector(scene_points)
        full_scene.colors = o3d.utility.Vector3dVector(scene_colors)
        full_scene = full_scene.voxel_down_sample(voxel_size=voxel_size)
        full_scene_list = [{'name':'RGB','group' : 'point_cloud', 'geometry':full_scene}]

        # Combined instance-colored point cloud
        labeled_mask = np.sum(instance_colors, axis=1) != 0
        labeled_scene = o3d.geometry.PointCloud()
        labeled_scene.points = o3d.utility.Vector3dVector(scene_points[labeled_mask])
        labeled_scene.colors = o3d.utility.Vector3dVector(instance_colors[labeled_mask])
        labeled_scene = labeled_scene.voxel_down_sample(voxel_size=voxel_size)
        labeled_scene_list = [{'name':'labeled_scene','group' : 'point_cloud', 'geometry':labeled_scene}]

        # Draw the scene
        vis.draw(instances_list+full_scene_list+labeled_scene_list, show_skybox=False, bg_color=(0,0,0,1))


    def __vis_one_object(self, point_ids, scene_points):
        points = scene_points[point_ids]
        color = (np.random.rand(3) * 0.7 + 0.3)
        colors = np.tile(color, (points.shape[0], 1))
        return point_ids, points, colors, color, np.mean(points, axis=0)

if __name__ == '__main__':
    seq_name = 'disordered_A_Bathroom-18516'
    root_env_dir = '/workspace/data/scenes'
    data_root_dir = '/workspace/data/scenes/disordered/A/Bathroom-18516/frames'

    scene_segmentation = SceneSegmentation(root_env_dir=root_env_dir,
                                            seq_name=seq_name, 
                                            save_root_path=data_root_dir)
    scene_segmentation.run()
    scene_segmentation.scene_visualize()