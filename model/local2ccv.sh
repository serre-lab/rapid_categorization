#!/bin/bash

rsync -avz --exclude-from=/home/sven2/python/hmax/levels/Xdata2cifs.txt -e ssh /media/sven2/data/sven2/AnNonAn/ seberhar@transfer2.ccv.brown.edu:/users/seberhar/hmax/levels/AnNonAn_cc256/
