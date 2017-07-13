#!/usr/bin/env bash

NAME=$(basename $(pwd)-training)
USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f ../..)

docker rm -f $NAME
rm snapshots/*
sudo mount --make-shared /nrs/turaga

echo "Starting as user ${USER_ID}"


NV_GPU=1 nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga \
    -v /nrs/turaga:/nrs/turaga:shared \
    -w ${PWD} \
    --name ${NAME} \
    funkey/gunpowder:v0.2 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH && python make_net.py && python -u train.py"

