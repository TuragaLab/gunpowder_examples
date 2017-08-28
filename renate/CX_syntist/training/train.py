from __future__ import print_function
import glob
import numpy as np
import math

import gunpowder
from gunpowder import *
from gunpowder.nodes import *
from gunpowder.caffe import Train, SolverParameters
import malis


def train(max_iteration, gpu, voxel_size):

    # get most recent training result
    solverstates = [ int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate') ]
    if len(solverstates) > 0:
        trained_until = max(solverstates)
        print("Resuming training from iteration " + str(trained_until))
    else:
        trained_until = 0
        print("Starting fresh training")
    if trained_until < phase_switch and max_iteration > phase_switch:
        # phase switch lies in-between, split training into to parts
        train(max_iteration=phase_switch, gpu=gpu, voxel_size=voxel_size)
        trained_until = phase_switch

    # switch from euclidean to malis after "phase_switch" iterations
    if max_iteration <= phase_switch:
        phase = 'euclid'
    else:
        phase = 'malis'
    print("Training until " + str(max_iteration) + " in phase " + phase)

    # define request
    request = BatchRequest()
    shape_input  = (132, 132, 132) * np.asarray(voxel_size)
    shape_output = (44, 44, 44) * np.asarray(voxel_size)
    request.add_volume_request(VolumeTypes.RAW, shape_input)
    request.add_volume_request(VolumeTypes.GT_LABELS, shape_output)
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, shape_output)
    if phase == 'malis':
        request.add_volume_request(VolumeTypes.MALIS_COMP_LABEL, shape_output)
    request.add_volume_request(VolumeTypes.LOSS_SCALE, shape_output)
    request.add_volume_request(VolumeTypes.PRED_AFFINITIES, shape_output)
    request.add_points_request(PointsTypes.PRESYN, shape_output)
    request.add_points_request(PointsTypes.POSTSYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_BM_POSTSYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN, shape_output)
    request.add_volume_request(VolumeTypes.LOSS_SCALE_BM_PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.LOSS_SCALE_BM_POSTSYN, shape_output)
    request.add_volume_request(VolumeTypes.PRED_BM_PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.PRED_BM_POSTSYN, shape_output)

    # define settings binary mask (rasterization of synapse points to binary mask)
    volumetypes_to_pointstype = {
                                 VolumeTypes.GT_BM_PRESYN: PointsTypes.PRESYN,
                                 VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN: PointsTypes.PRESYN,
                                 VolumeTypes.GT_BM_POSTSYN: PointsTypes.POSTSYN,
                                 VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN: PointsTypes.POSTSYN
                                }
    rastersetting_bm = RasterizationSetting(marker_size_physical=32)
    rastersetting_mask_syn = RasterizationSetting(marker_size_physical=96, donut_inner_radius=32, invert_map=True)
    volumetype_to_rastersettings = {
                                    VolumeTypes.GT_BM_PRESYN: rastersetting_bm,
                                    VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN: rastersetting_mask_syn,
                                    VolumeTypes.GT_BM_POSTSYN: rastersetting_bm,
                                    VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN: rastersetting_mask_syn,
                                   }

    # define names of datasets in snapshot file
    snapshot_dataset_names ={
                             VolumeTypes.RAW: 'volumes/raw',
                             VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                             VolumeTypes.GT_AFFINITIES: 'volumes/labels/affs',
                             VolumeTypes.PRED_AFFINITIES: 'volumes/predicted_affs',
                             VolumeTypes.LOSS_SCALE: 'volumes/loss_scale',
                             VolumeTypes.LOSS_GRADIENT: 'volumes/predicted_affs_loss_gradient',
                             VolumeTypes.GT_BM_PRESYN: 'volumes/labels/gt_bm_presyn',
                             VolumeTypes.GT_BM_POSTSYN: 'volumes/labels/gt_bm_postsyn',
                             VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN: 'volumes/labels/gt_mask_exclusivezone_presyn',
                             VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN: 'volumes/labels/gt_mask_exclusivezone_postsyn',
                             VolumeTypes.PRED_BM_PRESYN: 'volumes/predicted_bm_presyn',
                             VolumeTypes.PRED_BM_POSTSYN: 'volumes/predicted_bm_postsyn',
                             VolumeTypes.LOSS_SCALE_BM_PRESYN: 'volumes/loss_scale_presyn',
                             VolumeTypes.LOSS_SCALE_BM_POSTSYN: 'volumes/loss_scale_postsyn',
                            }

    # set network inputs, outputs and gradients
    train_inputs = {
                    VolumeTypes.RAW: 'data',
                    VolumeTypes.GT_AFFINITIES: 'aff_label',
                    VolumeTypes.GT_BM_PRESYN: 'bm_presyn_label',
                    VolumeTypes.GT_BM_POSTSYN:'bm_postsyn_label',
                    VolumeTypes.LOSS_SCALE: 'segm_scale',
                    VolumeTypes.LOSS_SCALE_BM_PRESYN: 'bm_presyn_scale',
                    VolumeTypes.LOSS_SCALE_BM_POSTSYN:'bm_postsyn_scale',
                    }
    if phase == 'malis':
        train_inputs[VolumeTypes.MALIS_COMP_LABEL] = 'comp_label'
        train_inputs['affinity_neighborhood'] = 'nhood'
    train_outputs = {
                     VolumeTypes.PRED_BM_PRESYN: 'bm_presyn_pred',
                     VolumeTypes.PRED_BM_POSTSYN: 'bm_postsyn_pred',
                     VolumeTypes.PRED_AFFINITIES: 'aff_pred',
                    }
    train_gradients = {
                       VolumeTypes.LOSS_GRADIENT_PRESYN: 'bm_presyn_pred',
                       VolumeTypes.LOSS_GRADIENT_POSTSYN: 'bm_postsyn_pred',
                       VolumeTypes.LOSS_GRADIENT: 'aff_pred',
                      }

    # set solver parameters
    solver_parameters = SolverParameters()
    solver_parameters.train_net       = 'net.prototxt'
    solver_parameters.base_lr         = 1e-4
    solver_parameters.momentum        = 0.99
    solver_parameters.momentum2       = 0.999
    solver_parameters.delta           = 1e-8
    solver_parameters.weight_decay    = 0.000005
    solver_parameters.lr_policy       = 'inv'
    solver_parameters.gamma           = 0.0001
    solver_parameters.power           = 0.75
    solver_parameters.snapshot        = 10000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type            = 'Adam'
    if trained_until > 0:
        solver_parameters.resume_from = 'net_iter_' + str(trained_until) + '.solverstate'
    else:
        solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage(phase)

    # set source of data
    data_sources = tuple(DvidSource(hostname='slowpoke2',
                        port=8000,
                        uuid='cb7dc',
                        volume_array_names = {VolumeTypes.RAW: 'grayscale',
                                              VolumeTypes.GT_LABELS: 'labels'},
                        points_array_names = {PointsTypes.PRESYN: 'combined_synapses_08302016',
                                              PointsTypes.POSTSYN: 'combined_synapses_08302016'},
                        points_rois = {PointsTypes.PRESYN: Roi(offset=(76000, 20000, 64000), shape=(4000, 4000, 16000)),
                                       PointsTypes.POSTSYN: Roi(offset=(76000, 20000, 64000), shape=(4000, 4000, 16000))},
                        points_voxel_size = {PointsTypes.PRESYN: voxel_size,
                                             PointsTypes.POSTSYN: voxel_size}) +
                    RandomLocation(focus_points_type=focus_points_type) +
                    Normalize()
                    for focus_points_type in (3*[PointsTypes.PRESYN] + 2*[None])
    )

    # define pipeline to process batches
    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        RasterizePoints(volumetypes_to_pointstype=volumetypes_to_pointstype, volumetypes_to_rastersettings=volumetype_to_rastersettings) +
        ElasticAugment(control_point_spacing=[40, 40, 40], jitter_sigma=[2, 2, 2], rotation_interval=[0, math.pi / 2.0],
                       prob_slip=0.01, prob_shift=0.01, max_misalign=1, subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(malis.mknhood3d()) +
        SplitAndRenumberSegmentationLabels() +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        PrepareMalis() +
        BalanceLabels(labels_to_loss_scale_volume={VolumeTypes.GT_BM_PRESYN: VolumeTypes.LOSS_SCALE_BM_PRESYN,
                                                   VolumeTypes.GT_BM_POSTSYN: VolumeTypes.LOSS_SCALE_BM_POSTSYN,
                                                   VolumeTypes.GT_LABELS: VolumeTypes.LOSS_SCALE},
                      labels_to_mask_volumes={VolumeTypes.GT_BM_PRESYN:  [VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN],
                                              VolumeTypes.GT_BM_POSTSYN: [VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN],}) +
        PreCache(cache_size=40, num_workers=10) +
        Train(solver_parameters, inputs=train_inputs, outputs=train_outputs, gradients=train_gradients, use_gpu=gpu) +
        Snapshot(dataset_names=snapshot_dataset_names, every=5000, output_filename='batch_{id}.hdf', compression_type="gzip")
    )

    print("Training for", max_iteration, "iterations")
    with build(batch_provider_tree) as minibatch_maker:
        for i in range(max_iteration):
            minibatch_maker.request_batch(request)
    print("Finished")


if __name__ == "__main__":
    gunpowder.set_verbose(False)

    phase_switch = 20000
    train(max_iteration=200000, gpu=0, voxel_size=(8,8,8))
