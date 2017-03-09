#!/usr/bin/env python

# Collecting true_labels and source_filenames for all batches

import os
import pickle
import rapid_categorization.model.util
import numpy as np

pred_pth = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/predictions'
source_out_fn = os.path.join(pred_pth, 'source.pickle')
batch_num = 60
label_source_file = 'setb50k'

# Collect predictions
source_info = dict()
for i in xrange(batch_num):
    true_labels, source_filenames = rapid_categorization.model.util.load_labels(label_source_file, i)
    if i == 0:
        source_info['true_labels'] = true_labels
        source_info['source_filenames'] = source_filenames
    else:
        source_info['true_labels'] = np.append(source_info['true_labels'], true_labels)
        source_info['source_filenames'] = np.append(source_info['source_filenames'], source_filenames)

with open(source_out_fn, 'wb') as fid:
    pickle.dump(source_info, fid)
