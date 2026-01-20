import os
from tqdm import tqdm
import yaml

from preprocess import Preprocess
from scene_segmentation import SceneSegmentation
from tasmap_pp_inference import TMPPInference
from utils.dataset import TMPPDataset
from utils.logger import TMPPLogger

# Load configuration
config_file_path = "/workspace/tasmap_pp/configs/tasmap_pp.yaml"
cfg = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)

root_env_dir = cfg["data_path"]["root_env_dir"]
if_root_env_dir = cfg["data_path"]["if_root_env_dir"]
os.makedirs(if_root_env_dir, exist_ok=True)
logger = TMPPLogger(cfg["data_path"]["log_file_path"])



dataset = TMPPDataset(root_env_dir=root_env_dir)
scene_list = dataset.get_scene_list()
pbar = tqdm(sorted(scene_list), desc="Processing Scenes", unit="scene")
for idx, (scene_id, house_id, room_id) in enumerate(pbar):

    seq_name = f"{scene_id}_{house_id}_{room_id}"
    logger.set_cur_log(seq_name)
    data_root_dir = os.path.join(root_env_dir, scene_id, house_id, room_id, 'frames')

    # Preprocessing
    if not logger.is_done("Step 1 - Preprocess"):
        logger.start("Step 1 - Preprocess")

        camera_config = cfg["camera_config"]
        Preprocess(preprocessed_root_path=data_root_dir,
                        save_root_path=data_root_dir,
                        camera_config=camera_config,
                        pcd_vis_flag=False,
                        seg_GT_flag=True)

        logger.stop("Step 1 - Preprocess")


    # Segmentation
    if not logger.is_done("Step 2 - 3DSeg"):
        logger.start("Step 2 - 3DSeg")
        scene_segmentation = SceneSegmentation(root_env_dir=root_env_dir,
                                               seq_name=seq_name, 
                                               save_root_path=data_root_dir)
        scene_segmentation.run()
        # scene_segmentation.scene_visualize()
        logger.stop("Step 2 - 3DSeg")


    # Inference
    tasmap_pp_results_dir = os.path.join(if_root_env_dir, scene_id, house_id, room_id, "tasmap_pp_results")
    if not logger.is_done("Step 3 - Inference"):
        logger.start("Step 3 - Inference")
        seg3D_path = data_root_dir

        inferencer = TMPPInference(root_env_dir=root_env_dir,
                                   seq_name=seq_name, 
                                    seg3D_path=data_root_dir,
                                    save_root_path=tasmap_pp_results_dir,
                                    model_name=cfg["open_ai"]["model_name"],
                                    api_key=cfg["open_ai"]["api_key"],
                                    debug=True)
        inferencer.run(max_workers=64)


        inferencer.semantic_fusion()
        inferencer.visualize()
        logger.stop("Step 3 - Inference")
            




                        