#!/usr/bin/bash

IMAGE_NAME="data_prep:latest"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# WORK_DIR="/workspace"
# SAVE_DIR="/home/$USER/project/workspace/rvt"

# docker run -it --rm --ipc=host\
#   --name data_prep \
#   -v "${SCRIPT_DIR}/..":"/home/$USER/src" \
#   -v "/home/$USER/datasets":/datasets \
#   ${IMAGE_NAME}

docker run -it --rm --ipc=host \
  --name data_prep \
  -v "${SCRIPT_DIR}/..":"/home/$USER/src" \
  -v "/home/jiayli/datasets":/datasets \
  ${IMAGE_NAME}