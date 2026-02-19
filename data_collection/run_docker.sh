#!/usr/bin/env bash
# sudo ./data_collection/run_docker.sh
set -e -o pipefail

ROOT_DIR=$( cd -- $(realpath "${BASH_SOURCE[0]}" | xargs dirname | xargs dirname) &> /dev/null && pwd )

WORKSPACE_PATH="$ROOT_DIR/data_collection"
TMPP_DATA_PATH="$ROOT_DIR/data"
DATA_PATH="$ROOT_DIR/og_data"

GUI=true


# # Move directories from their legacy paths.
# if [ -e "${DATA_PATH}/og_dataset" ]; then
#     mv "${DATA_PATH}/og_dataset" "${DATA_PATH}/datasets/og_dataset"
# fi
# if [ -e "${DATA_PATH}/assets" ]; then
#     mv "${DATA_PATH}/assets" "${DATA_PATH}/datasets/assets"
# fi

docker pull stanfordvl/omnigibson:1.0.0
DOCKER_DISPLAY=""
OMNIGIBSON_HEADLESS=1

xhost +local:root
DOCKER_DISPLAY=$DISPLAY
OMNIGIBSON_HEADLESS=0
WORKSPACE="workspace"

docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DOCKER_DISPLAY} \
    -e OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS} \
    -v $WORKSPACE_PATH:/workspace/data_collection \
    -v $TMPP_DATA_PATH:/workspace/data \
    -v $DATA_PATH:/data \
    --network=host --name data_collection -it stanfordvl/omnigibson:1.0.0 \
    bash -c "pip install -r /workspace/data_collection/requirements.txt && bash"

xhost -local:root
