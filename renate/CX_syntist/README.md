# Description
Scripts to train and process a network on pre- & postsynaptic location detection and membrane affinities prediction.

# gunpowder
* gunpowder from funkey/gunpowder branch: `release-v0.3` , commit: 5c3b484

# How to run ...
... training: 
1.) create .prototxt network file: define network in training/make_net.py, then run training/run_make_net.sh
2.) start training: set hyperparameters in train.py, then run training/run_training_docker.sh (to run it with nvidia-docker) or training/run_training_slurm.sh (to run it on the slurm cluster)

... processing:
1.) start inference: change path to trained weights in processing/process.py, then run processing/run_processing_docker.sh (to run it with nvidia-docker) or processing/run_processing_slurm.sh (to run it on the slurm cluster)


