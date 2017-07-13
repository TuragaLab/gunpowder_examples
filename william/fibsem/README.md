Run with:
 * gunpowder from TuragaLab/gunpowder @ commit 3ac377d, which has the `Typecast` gunpowder BatchFilter
 * `$ ./run_train.sh`, which requires nvidia-docker

Known issues:
 * You''' probably see an assertion error at least once with `gunpowder/__init__.pyc`. It might have to be changed to `__init__.py`, without `pyc`, for it to work. 
 * You might get an hdf5 read error for `groundtruth_seg.h5`. This happens ~ 1/3 of the time... unpredictably, from my perspective. Try re-running -- that fixes it for me!! (William)



