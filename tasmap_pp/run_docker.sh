#!/usr/bin/env bash
# sudo ./tasmap_pp/run_docker.sh
set -e -o pipefail

ROOT_DIR=$( cd -- $(realpath "${BASH_SOURCE[0]}" | xargs dirname | xargs dirname) &> /dev/null && pwd )
WORKSPACE_PATH="$ROOT_DIR/tasmap_pp"
DATA_PATH="$ROOT_DIR/data"
GUI=true


# Check if Docker image 'tasmap_plus_plus' exists
if ! docker image inspect tasmap_plus_plus >/dev/null 2>&1; then
    echo "[INFO] Docker image 'tasmap_plus_plus' not found. Building now..."
    docker build -t tasmap_plus_plus tasmap_pp/.
else
    echo "[INFO] Docker image 'tasmap_plus_plus' already exists. Skipping build."
fi

docker run --name tasmap_plus_plus -it --privileged \
    --gpus all \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $WORKSPACE_PATH:/workspace/tasmap_pp\
    -v $DATA_PATH:/workspace/data/\
    -e DISPLAY=$DISPLAY \
    --net host \
    tasmap_plus_plus /bin/bash

if [ "$GUI" = true ] ; then
    xhost -local:root
fi