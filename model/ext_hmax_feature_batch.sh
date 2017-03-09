#!/bin/bash

source ~/.modules
source ~/.bashrc
#export PYTHONPATH=/users/seberhar/:$PYTHONPATH
PYTHONPATH=/users/seberhar/:$PYTHONPATH
source venv/bin/activate --no-site-packages
LAYER=${1:-c2}
DICT=$2
ID=${SLURM_ARRAY_TASK_ID:-$3}
NUM_PARALLEL=8
ID_START=$((ID*NUM_PARALLEL))
for RUN in $(seq 0 $((NUM_PARALLEL-1))); do
	RUN_ID=$((ID*NUM_PARALLEL+RUN))
	echo ./ext_hmax_features.py $LAYER $RUN_ID $RUN $DICT
	./ext_hmax_features.py $LAYER $RUN_ID $RUN $DICT &
done
wait # for jobs to finish

Inception Image Classification Model