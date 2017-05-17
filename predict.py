import os

import gunpowder
from gunpowder import Coordinate, Hdf5Source, Normalize, Padding, Snapshot, PrintProfilingStats, Chunk, BatchSpec
from gunpowder.caffe import Predict


def predict():
    iteration = 18000
    prototxt = '/groups/turaga/home/grisaitisw/src/syntist/gunpowder/tasks/training/net_test.prototxt'
    weights = os.path.join('/groups/turaga/home/grisaitisw/experiments/20170524/090000', 'net_iter_%d.caffemodel' % iteration)

    input_size = Coordinate((236, 236, 236))
    output_size = Coordinate((148, 148, 148))

    pipeline = (
        Hdf5Source(
            '/groups/turaga/home/grisaitisw/src/gunpowder/examples/fib25/tstvol-520-1-h5_y0_x0_xy0_angle000.0.h5',
            raw_dataset='volumes/raw',
            resolution=(8, 8, 8),
        ) +
        Normalize() +
        Padding() +
        Predict(prototxt, weights, use_gpu=0) +
        Snapshot(
            every=1,
            output_dir=os.path.join('chunks', '%d' % iteration),
            output_filename='chunk_{id}.hdf',
            with_timestamp=True
        ) +
        PrintProfilingStats() +
        Chunk(
            BatchSpec(
                input_size,
                output_size,
                resolution=(8, 8, 8),
            )
        ) +
        Snapshot(
            every=1,
            output_dir=os.path.join('processed', '%d' % iteration),
            output_filename='tstvol-520-1-h5_2_{id}.hdf'
        )
    )

    # request a "batch" of the size of the whole dataset
    with gunpowder.build(pipeline) as p:
        shape = p.get_spec().roi.get_shape()
        p.request_batch(
            BatchSpec(
                shape,
                shape - (input_size - output_size),
                resolution=(8, 8, 8),
            )
        )


if __name__ == "__main__":
    predict()
