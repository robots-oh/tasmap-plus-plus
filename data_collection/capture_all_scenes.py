import os
import json
import shutil
from zoneinfo import ZoneInfo
from datetime import datetime
import time

from tqdm import tqdm



scene_list = []
env_root_dir = '/workspace/data/scenes'
for scene_id in os.listdir(env_root_dir):
    if os.path.isfile(os.path.join(env_root_dir, scene_id)):
        continue
    for house_id in os.listdir(os.path.join(env_root_dir, scene_id)):
        for room_id in os.listdir(os.path.join(env_root_dir, scene_id, house_id)):
            scene_list.append((scene_id, house_id, room_id))
total_scenes = len(scene_list)

log_file = os.path.join(env_root_dir, "capture_log.json")
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump({}, f, indent=4)

pbar = tqdm(sorted(scene_list), desc="Processing Scenes", unit="scene")
for idx, (scene_id, house_id, room_id) in enumerate(pbar):
    scene_dir_path = os.path.join(env_root_dir, scene_id, house_id, room_id)
    current_scene = f'{scene_id}_{house_id}_{room_id}'
    with open(log_file, "r") as f:
        log_dict = json.load(f)
    is_success = log_dict.get(current_scene, {}).get("status") == "success"
    if is_success:
        tqdm.write(f"[{idx+1}/{total_scenes}] Skipped scene '{scene_id}' (already successful)")
        continue

    tqdm.write(f"[{idx+1}/{total_scenes}] Processing scene {current_scene}")
    if os.path.isdir(os.path.join(scene_dir_path, 'frames')):
        shutil.rmtree(os.path.join(scene_dir_path, 'frames'))

    start_time = time.time()
    exit_code = os.system(f'OMNIGIBSON_HEADLESS=1 python /workspace/data_collection/capture_single_room_traj.py --scene_dir {scene_dir_path} --save_seg_info True > /dev/null 2>&1' )
    end_time = time.time()
    duration = end_time - start_time
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    if exit_code == 0:
        status = "success"
    else:
        status = f"fail{exit_code}"

    log_dict[current_scene] = {
                    "idx": idx,
                    "scene": current_scene,
                    "status": status,
                    "duration": duration,
                    "created": f"{now_str}",
                }

    tqdm.write(f"[{idx+1}/{total_scenes}] {status} | {duration} | {current_scene}")
    with open(log_file, "w") as f:
        json.dump(log_dict, f, indent=4)




