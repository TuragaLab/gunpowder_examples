from __future__ import print_function

import math
import os

import h5py
import malis
import numpy as np

import gunpowder

imported_gunpowder = os.path.dirname(gunpowder.__file__)
desired_gunpowder = os.path.abspath("./gunpowder/gunpowder")
assert imported_gunpowder == desired_gunpowder, "Imported gunpowder at {}, not at {}".format(imported_gunpowder, desired_gunpowder)

from gunpowder import VolumeTypes, RandomLocation, Normalize, RandomProvider, GrowBoundary, \
    SplitAndRenumberSegmentationLabels, AddGtAffinities, PreCache, Snapshot, BatchRequest, ElasticAugment, \
    SimpleAugment, IntensityAugment, BalanceAffinityLabels, PrintProfilingStats, Typecast, Reject
from gunpowder.caffe import Train
from gunpowder.nodes.dvid_source import DvidSource

import constants


def train():

    gunpowder.set_verbose(False)

    affinity_neighborhood = malis.mknhood3d()
    solver_parameters = gunpowder.caffe.SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 1e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 10000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    request = BatchRequest()
    request.add_volume_request(VolumeTypes.RAW, constants.input_shape)
    request.add_volume_request(VolumeTypes.GT_LABELS, constants.output_shape)
    request.add_volume_request(VolumeTypes.GT_MASK, constants.output_shape)
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, constants.output_shape)
    request.add_volume_request(VolumeTypes.LOSS_SCALE, constants.output_shape)

    data_providers = list()
    fibsem_dir = "/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col"
    for volume_name in ("tstvol-520-1-h5",):
        h5_filepath = "./{}.h5".format(volume_name)
        path_to_labels = os.path.join(fibsem_dir, volume_name, "groundtruth_seg.h5")
        with h5py.File(path_to_labels, "r") as f_labels:
            mask_shape = f_labels["main"].shape
        with h5py.File(h5_filepath, "w") as h5:
            h5['volumes/raw'] = h5py.ExternalLink(os.path.join(fibsem_dir, volume_name, "im_uint8.h5"), "main")
            h5['volumes/labels/neuron_ids'] = h5py.ExternalLink(path_to_labels, "main")
            h5.create_dataset(
                name="volumes/labels/mask",
                dtype="uint8",
                shape=mask_shape,
                fillvalue=1,
            )
        data_providers.append(
            gunpowder.Hdf5Source(
                h5_filepath,
                datasets={
                    VolumeTypes.RAW: 'volumes/raw',
                    VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                    VolumeTypes.GT_MASK: 'volumes/labels/mask',
                },
                resolution=(8, 8, 8),
            )
        )
    dvid_source = DvidSource(
        hostname='slowpoke3',
        port=32788,
        uuid='341',
        raw_array_name='grayscale',
        gt_array_name='groundtruth',
        gt_mask_roi_name="seven_column_eroded7_z_lt_5024",
        resolution=(8, 8, 8),
    )
    data_providers.extend([dvid_source])
    data_providers = tuple(
        provider +
        RandomLocation() +
        Reject(min_masked=0.5) +
        Normalize()
        for provider in data_providers
    )

    # create a batch provider by concatenation of filters
    batch_provider = (
        data_providers +
        RandomProvider() +
        ElasticAugment([20, 20, 20], [0, 0, 0], [0, math.pi / 2.0]) +
        SimpleAugment(transpose_only_xy=False) +
        GrowBoundary(steps=2, only_xy=False) +
        AddGtAffinities(affinity_neighborhood) +
        BalanceAffinityLabels() +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=False) +
        PreCache(
            request,
            cache_size=11,
            num_workers=10) +
        Train(solver_parameters, use_gpu=0) +
        Typecast(volume_dtypes={
            VolumeTypes.GT_LABELS: np.dtype("uint32"),
            VolumeTypes.GT_MASK: np.dtype("uint8"),
            VolumeTypes.LOSS_SCALE: np.dtype("float32"),
        }, safe=True) +
        Snapshot(every=50, output_filename='batch_{id}.hdf') +
        PrintProfilingStats(every=50)
    )

    n = 500000
    print("Training for", n, "iterations")

    with gunpowder.build(batch_provider) as pipeline:
        for i in range(n):
            pipeline.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()
