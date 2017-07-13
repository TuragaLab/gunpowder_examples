Run with:
 * gunpowder from TuragaLab/gunpowder @ commit 3ac377d, which has the `Typecast` gunpowder BatchFilter
 * `$ ./run_train.sh`, which requires nvidia-docker

Known issues:
 * You might get an hdf5 read error for `groundtruth_seg.h5`. This happens ~ 1/3 of the time... unpredictably, from my perspective. Try re-running -- that fixes it for me!! (William)



