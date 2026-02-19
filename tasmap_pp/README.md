# Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark

This folder contain the **execution pipeline** for TASMap++.

## Installation
```bash
# create docker environment
sudo ./run_docker.sh
```

## Dataset
Please see [README](https://github.com/robots-oh/tasmap-plus-plus/tree/main/data_collection) in `data_collection` folder.

## Usage
```bash
# start docker environment
docker start tasmap_plus_plus
docker attach tasmap_plus_plus

# run TASMap++
python tasmap_pp/run.py

# run TASMap++ (multi-threaded)
python tasmap_pp/run.py --max_workers 64
```



