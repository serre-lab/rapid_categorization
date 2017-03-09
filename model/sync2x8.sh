#!/bin/bash

TARGET=/media/data/nsf_levels
SOURCE=seberhar@tccv:/users/seberhar/data/data/AnNonAnNIPS

rsync -avz -e ssh $SOURCEC/predictions $TARGET/predictions

