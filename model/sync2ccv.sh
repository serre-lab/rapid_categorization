#!/bin/bash

SOURCE=/media/data/nsf_levels
SOURCEC=/media/data_cifs/nsf_levels
TARGET=seberhar@tccv:/users/seberhar/data/data/AnNonAnNIPS

#rsync -avz -e ssh $SOURCEC/TURK_IMS $TARGET/raw_ims
#rsync -avz -e ssh $SOURCEC/raw_ims $TARGET/raw_ims
scp $SOURCE/*.caffemodel $TARGET/
rsync -avz -e ssh $SOURCE/ $TARGET/

