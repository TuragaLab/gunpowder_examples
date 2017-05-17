from __future__ import print_function

import glob
import os

import h5py
import neuroglancer
import time

from neuroglancer.server import global_server_args

from run import target_h5_path, target_affinities_h5_key, source_affinities, source_affinity_superslice, source_location

for snapshot_h5 in glob.glob("./snapshots/*.hdf"):
    print(snapshot_h5)


# affinities_subvolume = h5py.File(target_h5_path)[target_affinities_h5_key][:, 0:100, 0:1000, 0:1000]

neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
global_server_args.update(dict(bind_address='0.0.0.0'))

print(neuroglancer.server.global_server_args)

viewer = neuroglancer.Viewer(voxel_size=[8, 8, 8])
shader_affinity = """
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
"""
# viewer.add(
#     affinities_subvolume,
#     name='affinities_subvolume',
#     # offset=tuple(reversed(source_location)),
#     offset=(2800 * 8, 2200 * 8, 6000 * 8),
#     shader=shader_affinity
# )
viewer.add(
    source_affinities,
    name='source_affinities',
    shader=shader_affinity
)
print(viewer)

time.sleep(999999999)
