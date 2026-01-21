import os
import argparse
from tqdm import tqdm
import yaml

from preprocess import Preprocess
from scene_segmentation import SceneSegmentation
from tasmap_pp_inference import TMPPInference
from utils.dataset import TMPPDataset
from utils.logger import TMPPLogger


def parse_args():
    parser = argparse.ArgumentParser(description="TasMap++ pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/tasmap_pp/configs/tasmap_pp.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Max number of workers for inference",
    )
    parser.add_argument(
        "--debug",
        default=True,
        help="Enable debug mode for inference (overrides cfg if needed)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

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
                                        debug=args.debug)
            inferencer.run(max_workers=args.max_workers)


            inferencer.semantic_fusion()
            inferencer.visualize()
            logger.stop("Step 3 - Inference")
            


if __name__ == "__main__":
    main()

                        