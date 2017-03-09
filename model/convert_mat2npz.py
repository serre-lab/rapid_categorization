#!/usr/bin/env python

# .mat data files to .npz for python


import sys
import os
import scipy.io as sio
import numpy as np
import rapid_categorization.model.util

if len(sys.argv) < 2:
    print 'No filenames provided.'
    exit()

for input_name in sys.argv[1:]:
    if not os.path.isfile(input_name):
        print 'File not found: %s' % input_name
        continue
    output_name = input_name.replace('.mat', '.npz')
    if os.path.isfile(output_name):
        print 'Already done: %s' % output_name
        continue
    print '%s -> %s' % (input_name, output_name)
    mat_data = sio.loadmat(input_name)
    if not 'data' in mat_data:
        print 'No data in file: %s' % input_name
        continue
    np.savez(output_name, data=mat_data['data'])
