import numpy as np
from gunpowder import *
from gunpowder.caffe import *
import malis


def predict(roi_synapses, voxel_size):

    # get network and weights to use for inference
    prototxt = 'net.prototxt'
    weights  = 'net_iter_200000.caffemodel'

    # create template for chunk
    chunk_spec_template = BatchRequest()
    shape_input_template  = [132, 132, 132] * np.asarray(voxel_size)
    shape_output_template = [44, 44, 44] * np.asarray(voxel_size)
    chunk_spec_template.add_volume_request(VolumeTypes.RAW, shape_input_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_LABELS, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_AFFINITIES, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.PRED_AFFINITIES, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.PRED_BM_PRESYN, shape_output_template)
    chunk_spec_template.add_points_request(PointsTypes.PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_BM_POSTSYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.PRED_BM_POSTSYN, shape_output_template)
    chunk_spec_template.add_points_request(PointsTypes.POSTSYN, shape_output_template)

    # create batch request, shapes and Type of requests
    request = BatchRequest()
    shape_outputs = roi_synapses.get_shape()
    request.add_volume_request(VolumeTypes.RAW, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_LABELS, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, shape_outputs)
    request.add_volume_request(VolumeTypes.PRED_AFFINITIES, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.PRED_BM_PRESYN, shape_outputs)
    request.add_points_request(PointsTypes.PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_BM_POSTSYN, shape_outputs)
    request.add_volume_request(VolumeTypes.PRED_BM_POSTSYN, shape_outputs)
    request.add_points_request(PointsTypes.POSTSYN, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_POSTSYN, shape_outputs)

    # shift request roi to correct offset
    request_offset = roi_synapses.get_offset()
    for request_type in [request.volumes, request.points]:
        for type in request_type:
            request_type[type] += request_offset

    # set network inputs, outputs and resolutions of output volumes
    net_inputs = {VolumeTypes.RAW: 'data'}
    net_outputs = {
                   VolumeTypes.PRED_BM_PRESYN:'bm_presyn_pred',
                   VolumeTypes.PRED_BM_POSTSYN: 'bm_postsyn_pred',
                   VolumeTypes.PRED_AFFINITIES: 'aff_pred'
                  }
    output_resolutions = {
                          VolumeTypes.PRED_BM_PRESYN: voxel_size,
                          VolumeTypes.PRED_BM_POSTSYN: voxel_size,
                          VolumeTypes.PRED_AFFINITIES: voxel_size
                         }

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

    # set source of data
    data_source = (DvidSource(hostname='slowpoke2',
                               port=8000,
                               uuid='cb7dc',
                               volume_array_names={VolumeTypes.RAW: 'grayscale',
                                                   VolumeTypes.GT_LABELS: 'labels'},
                               points_array_names={PointsTypes.PRESYN: 'combined_synapses_08302016',
                                                   PointsTypes.POSTSYN: 'combined_synapses_08302016'},
                               points_rois={PointsTypes.PRESYN: roi_synapses.grow((0,0,0), Coordinate((352, 352, 352))),
                                            PointsTypes.POSTSYN: roi_synapses.grow((0,0,0), Coordinate((352, 352, 352)))},
                               points_voxel_size={PointsTypes.PRESYN:  voxel_size,
                                                  PointsTypes.POSTSYN: voxel_size}) +
                   Pad({VolumeTypes.RAW: Coordinate((704, 704, 704))},
                       {VolumeTypes.RAW: 255}) +
                   Normalize())

    # define pipeline to process chunk
    batch_provider_tree = (
            data_source +
            RasterizePoints(volumetypes_to_pointstype=volumetypes_to_pointstype,
                            volumetypes_to_rastersettings=volumetype_to_rastersettings) +
            AddGtAffinities(malis.mknhood3d()) +
            IntensityScaleShift(2, -1) +
            ZeroOutConstSections() +
            Predict(prototxt, weights, net_inputs, net_outputs, output_resolutions, use_gpu=0) +
            Chunk(chunk_spec_template, num_workers=1) +
            Snapshot(dataset_names=snapshot_dataset_names, every=1, output_filename='output', compression_type="gzip"))

    # request a "batch" of the size of the whole dataset
    with build(batch_provider_tree) as minibatch_maker:
        minibatch_maker.request_batch(request)
    print("Finished")


if __name__ == "__main__":

    predict(roi_synapses=Roi(offset=(80000, 20000, 64000), shape=(4000, 4000, 16000)),
            voxel_size=(8,8,8))
