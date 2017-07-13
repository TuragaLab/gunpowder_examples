#!/usr/bin/env bash

rm snapshots/*

NAME=$(basename "$PWD-prediction-2")

sudo mount --make-shared /nrs/turaga

docker rm -f $NAME

USER_ID=${UID}

echo "Starting as user ${USER_ID} with home ${HOME}"

NV_GPU=2 nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    -w ${PWD} \
    --name ${NAME} \
    funkey/gunpowder:latest \
    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH && python -u predict.py"

#    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH bash"
#    -e HOME=${USER_HOME} \
