#!/usr/bin/env bash

rm snapshots/*

export NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

NV_GPU=0 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /raid/julia/src/gunpowder:/opt/gunpowder \
    -v /raid/julia/data/cremi2017officialWebsite/:/cremidatasets/ \
    -w /workspace \
    --name $NAME \
    julia/gunpowder \
    bash -c 'PYTHONPATH=/opt/gunpowder/:$PYTHONPATH; python -u train.py'

