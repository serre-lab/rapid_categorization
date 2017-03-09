#!/bin/bash
# Run caffe on image set

WEIGHTS=/home/sven2/s2caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
MODELFN=train.prototxt
LAYER=fc7r
OUTPUTDB=/media/sven2/data/sven2/sets/all_fc7.lmdb
NUMIMG=5093303
BATCHSIZE=256
NUMBATCHES=$(( (NUMIMG-1) / BATCHSIZE + 1 ))
IGPU=0
/home/sven2/s2caffe/.build_release/tools/extract_features $WEIGHTS $MODELFN $LAYER $OUTPUTDB $NUMBATCHES lmdb GPU $IGPU
