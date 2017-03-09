#!/usr/bin/env python

# Pickle too slow. Let's convert stuff to something faster

import sys
import os
import pickle
import numpy as np
import rapid_categorization.levels.util

if len(sys.argv) < 2:
    print 'No filenames provided.'
    exit()

for input_name in sys.argv[1:]:
    if not os.path.isfile(input_name):
        print 'File not found: %s' % input_name
        continue
    output_name = input_name.replace('.pickle', '.npz')
    if os.path.isfile(output_name):
        print 'Already done: %s' % output_name
        continue
    print '%s -> %s' % (input_name, output_name)
    with open(input_name, 'rb') as fid:
        data = pickle.load(fid)
    np.savez(output_name, data=data)
