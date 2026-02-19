# Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark

This folder contains the **data collection** code for TASMap++.
Please note that a portion of the dataset has not been released due to licensing restrictions.

## Installation
```bash
# create docker environment
sudo ./data_collection/run_docker.sh
```

## Usage
```bash
# start docker environment
docker start data_collection
docker attach data_collection

# load scene
python /workspace/data_collection/scene_load.py

# capture single scene
python /workspace/data_collection/capture_single_room_traj.py

# capture all scenes
python /workspace/data_collection/capture_all_scenes.py
```

