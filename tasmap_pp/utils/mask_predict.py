import os
import sys
import glob
import warnings
import multiprocessing as mp
from tqdm import tqdm

sys.path.append('/workspace/third_party/detectron2/projects/CropFormer/')
sys.path.append('/workspace/third_party/detectron2/projects/CropFormer/demo_cropformer')

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
warnings.filterwarnings("ignore", category=UserWarning)

class MaskPredictor():
    def __init__(self,
                 seg_model_config_file='/workspace/third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml',
                 seg_model_opts=["MODEL.WEIGHTS", "/workspace/tasmap_pp/weights/Mask2Former_hornet_3x_576d0b.pth"]):
        self.confidence_threshold = 0.5
        self.mask_minimum_area = 400

        cfg = self.__seg_model_setup_cfg(seg_model_config_file, seg_model_opts)
        self.demo = VisualizationDemo(cfg)


    def __seg_model_setup_cfg(self, config_file, opts):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()
        return cfg


    def __create_colormap(self):
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)
        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap


    def __debug_save(self, origin_image, save_root_path, image_path, mask_image):
        colormap = self.__create_colormap()

        color_segmentation = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype=np.uint8)
        mask_ids = np.unique(mask_image)
        mask_ids.sort()

        text_list, text_center_list = [], []
        for mask_id in mask_ids:
            if mask_id == 0:
                continue
            color_segmentation[mask_image == mask_id] = colormap[mask_id]
            mask_pos = np.where(mask_image == mask_id)
            mask_center = (int(np.mean(mask_pos[1])), int(np.mean(mask_pos[0])))
            text_list.append(str(mask_id))
            text_center_list.append(mask_center)

        for text, text_center in zip(text_list, text_center_list):
            cv2.putText(color_segmentation, text, text_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        concatenate_image = np.concatenate((origin_image, color_segmentation), axis=1)
        concatenate_image = cv2.resize(concatenate_image, (concatenate_image.shape[1] // 2, concatenate_image.shape[0] // 2))
        debug_dir = os.path.join(save_root_path, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, os.path.basename(image_path).split('.')[0] + '.png'), concatenate_image)


    def __call__(self, data_root_path, save_root_path=None, debug_image_save_flag=False):
        if save_root_path is None:
            save_root_path = os.path.join(data_root_path, 'mask')
        os.makedirs(save_root_path, exist_ok=True)

        image_list = sorted(glob.glob(os.path.join(data_root_path, 'color/*.jpg')))
        for image_path in tqdm(image_list):
            # use PIL, to be consistent with evaluation
            origin_image = cv2.imread(image_path)
            predictions = self.demo.run_on_image(origin_image)

            ##### color_mask
            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores
            
            # select by confidence threshold
            selected_indexes = (pred_scores >= self.confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks  = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            mask_image = np.zeros((m_H, m_W), dtype=np.uint8)

            # rank
            mask_id = 1
            selected_scores, ranks = torch.sort(selected_scores)
            for index in ranks:
                num_pixels = torch.sum(selected_masks[index])
                if num_pixels < self.mask_minimum_area:
                    continue
                mask_image[(selected_masks[index]==1).cpu().numpy()] = mask_id
                mask_id += 1
            cv2.imwrite(os.path.join(save_root_path, os.path.basename(image_path).split('.')[0] + '.png'), mask_image)

            if debug_image_save_flag:
                self.__debug_save(origin_image, save_root_path, image_path, mask_image)