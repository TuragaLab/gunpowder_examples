from __future__ import print_function
import math

import caffe
from caffe import layers as L
from caffe import metalayers as ML

# Start a network
net = caffe.NetSpec()
input_shape  = [132, 132, 132]
output_shape = [44, 44, 44]

source_layers, max_shapes = [], []

# Data input layer
net.data = L.MemoryData(dim=[1, 1], ntop=1)
source_layers.append(net.data)
max_shapes.append(input_shape)

# Label input layer
net.aff_label = L.MemoryData(dim=[1, 3], ntop=1, include=[dict(phase=0)])
source_layers.append(net.aff_label)
max_shapes.append(output_shape)
net.bm_presyn_label = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])
source_layers.append(net.bm_presyn_label)
max_shapes.append(output_shape)
net.bm_postsyn_label = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])
source_layers.append(net.bm_postsyn_label)
max_shapes.append(output_shape)

# Components label layer
net.comp_label = L.MemoryData(dim=[1, 2], ntop=1, include=[dict(phase=0, stage='malis')])
source_layers.append(net.comp_label)
max_shapes.append(output_shape)

# Scale input layer
net.segm_scale = L.MemoryData(dim=[1, 3], ntop=1, include=[dict(phase=0, stage='euclid')])
source_layers.append(net.segm_scale)
max_shapes.append(output_shape)
net.bm_presyn_scale = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)]) # , stage='euclid')])
source_layers.append(net.bm_presyn_scale)
max_shapes.append(output_shape)
net.bm_postsyn_scale = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])  # , stage='euclid')])
source_layers.append(net.bm_postsyn_scale)
max_shapes.append(output_shape)

# Silence the not needed data and label integer values
net.nhood = L.MemoryData(dim=[1, 1, 3, 3], ntop=1, include=[dict(phase=0, stage='malis')])
source_layers.append(net.nhood)
max_shapes.append(output_shape)


net.unet = ML.UNet(net.data,
                   fmap_start=12,
                   depth=3,
                   fmap_inc_rule = lambda fmaps: int(math.ceil(float(fmaps) * 3)),
                   fmap_dec_rule = lambda fmaps: int(math.ceil(float(fmaps) / 3)),
                   downsampling_strategy = [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                   dropout = 0.0,
                   use_deconv_uppath=False,
                   use_stable_upconv=True)
net.aff_out = L.Convolution(net.unet, kernel_size=[1], num_output=3, param=[dict(lr_mult=1), dict(lr_mult=2)],
                            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),)
net.bm_presyn_out = L.Convolution(net.unet, kernel_size=[1], num_output=1, param=[dict(lr_mult=1), dict(lr_mult=2)],
                            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
net.bm_postsyn_out = L.Convolution(net.unet, kernel_size=[1], num_output=1, param=[dict(lr_mult=1), dict(lr_mult=2)],
                            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

# Choose output activation functions
net.aff_pred = L.Sigmoid(net.aff_out, ntop=1, in_place=False)
net.bm_presyn_pred = L.Sigmoid(net.bm_presyn_out, ntop=1, in_place=False)
net.bm_postsyn_pred = L.Sigmoid(net.bm_postsyn_out, ntop=1, in_place=False)

# Choose a loss function and input data, label and scale inputs. Only include it during the training phase (phase = 0)
net.euclid_loss = L.EuclideanLoss(net.aff_pred, net.aff_label, net.segm_scale,
                                    ntop=0, loss_weight=1.0, include=[dict(phase=0, stage='euclid')],)
net.malis_loss = L.MalisLoss(net.aff_pred, net.aff_label, net.comp_label, net.nhood,
                                    ntop=0, loss_weight=1.0, include=[dict(phase=0, stage='malis')],)
net.bm_presyn_euclid_loss = L.EuclideanLoss(net.bm_presyn_pred, net.bm_presyn_label, net.bm_presyn_scale,
                                    ntop=0, loss_weight=1.0, include=[dict(phase=0)])
net.bm_postsyn_euclid_loss = L.EuclideanLoss(net.bm_postsyn_pred, net.bm_postsyn_label, net.bm_postsyn_scale,
                                    ntop=0, loss_weight=1.0, include=[dict(phase=0)])

# Fix the spatial input dimensions. Note that only spatial dimensions get modified, the minibatch size
# and the channels/feature maps must be set correctly by the user (since this code can definitely not
# figure out the user's intent). If the code does not seem to terminate, then the issue is most likely
# a wrong number of feature maps / channels in either the MemoryData-layers or the network output.

# This function takes as input:
# - The network
# - A list of other inputs to test (note: the nhood input is static and not spatially testable, thus excluded here)
# - A list of the maximal shapes for each input
# - A list of spatial dependencies; here [-1, 0] means the Y axis is a free parameter, and the X axis should be identical to the Y axis.

source_layers = [net.data, net.aff_label, net.bm_presyn_label, net.bm_postsyn_label,
                    net.comp_label, net.segm_scale, net.bm_presyn_scale, net.bm_postsyn_scale, net.nhood]
max_shapes = [input_shape] + 8*[output_shape]

caffe.fix_input_dims(net=net, source_layers=source_layers, max_shapes=max_shapes, shape_coupled=[-1, -1, 0, 0, 0])

protonet = net.to_proto()
protonet.name = 'net'

# Store the network as prototxt
with open(protonet.name + '.prototxt', 'w') as f:
    print(protonet, file=f)