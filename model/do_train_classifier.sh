#!/bin/bash

source ~/.modules
source ~/.bashrc
PYTHONPATH=/users/seberhar/:$PYTHONPATH
source venv/bin/activate --no-site-packages
python train_classifier.py $1
