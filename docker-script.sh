#!/bin/bash

#  --gpus \"device=$1\" \
#  --gpus all \

docker run -it \
  --rm \
  --gpus \"device=$1\" \
  -v `pwd`:`pwd` \
  -w `pwd` \
  --net host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_HOME=/home/wingated/hfmodels \
  nsyr


#  pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel /bin/bash
