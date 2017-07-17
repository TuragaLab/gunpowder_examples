from __future__ import print_function

import numpy as np
import os
import math

from gunpowder import *
from gunpowder.nodes import *
from gunpowder.caffe import Train, SolverParameters

from customized_nodes import CremiSource, TrainSyntist


def train(max_iteration):

    # define a batch request
    request = BatchRequest()
    request.add_volume_request(VolumeTypes.RAW, (84, 268, 268))
    request.add_points_request(PointsTypes.PRESYN, (56, 56, 56))
    request.add_volume_request(VolumeTypes.GT_BM_PRESYN, (56, 56, 56))
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, (56, 56, 56))


    # define solver parameters
    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 1e-6
    solver_parameters.momentum = 0.99  # 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 100
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    # define where to find data
    data_dir = '/cremidatasets/'
    samples = ['sample_A_padded_20160501', 'sample_B_padded_20160501', 'sample_C_padded_20160501']
    volume_phys_offset = [(1520, 3644, 3644), (1480, 3644, 3644), (1480, 3644, 3644)]
    synapse_shapes = [(125, 1250, 1250), (125, 1250, 1250),
                      (60, 1250, 1250)]
    voxel_size = np.array([40, 4, 4])

    data_sources = tuple(
        CremiSource(
            os.path.join(data_dir, sample + '.hdf'),
            datasets={
                VolumeTypes.RAW: 'volumes/raw',
            },
            points_types=[PointsTypes.PRESYN],
            points_rois={PointsTypes.PRESYN: Roi(offset=tuple(np.array(volume_phys_offset[ii])/voxel_size),
                                                shape=synapse_shapes[ii])},
            volume_phys_offset={VolumeTypes.GT_LABELS: volume_phys_offset[ii]},
        ) +
        RandomLocation(focus_points_type=PointsTypes.PRESYN) +
        Normalize()
        for ii, sample in enumerate(samples)
    )

    # define pipeline to process batches
    pointstype_to_volumetypes = {
        PointsTypes.PRESYN: VolumeTypes.GT_BM_PRESYN,
    }

    cache_size, num_workers, sp_every = 40, 15, 10
    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        AddGtBinaryMapOfPoints(pointstype_to_volumetypes) +
        AddGtMaskExclusiveZone() +
        SimpleAugment(transpose_only_xy=True) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4, 40, 40], [0, 2, 2], [0, math.pi / 2.0], prob_slip=0.05, prob_shift=0.05, max_misalign=10,
                       subsample=8) +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        PreCache(request, cache_size, num_workers) +
        TrainSyntist(solver_parameters, use_gpu=0) +
        Snapshot(every=sp_every, output_filename='batch_{id}.hdf',
                 compression_type="gzip")
    )

    print("Training for", max_iteration, "iterations")
    with build(batch_provider_tree) as minibatch_maker:
        for i in range(max_iteration):
            minibatch_maker.request_batch(request)
    print("Finished")

if __name__ == "__main__":
    set_verbose(False)
    train(max_iteration=200000)
