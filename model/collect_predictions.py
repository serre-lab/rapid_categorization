#!/usr/bin/env python

# All predictions collected into one file

import numpy as np
import glob
import os

pred_pth = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/predictions'
pred_pth = '/media/sven2/data/sven2/AnNonAn/predictions'
out_fns = os.path.join(pred_pth, '%s_%s.npz')
all_pred_pths = glob.glob(pred_pth + '/*')
all_pred = [os.path.basename(p) for p in all_pred_pths]
batch_size = 5000
collect_items = ['pred_labels', 'hyper_dist']

force_rewrite = False

# Collect predictions
source_info = dict()
for pred, pred_pth in zip(all_pred, all_pred_pths):
    n_files = len(glob.glob(pred_pth + '/*.npz'))
    if n_files <= 0:
        continue
    print '%05d files for %s in %s...' % (n_files, pred, pred_pth)
    if not force_rewrite:
        if os.path.isfile(out_fns % (pred, collect_items[0])):
            print '   skipping.'
            continue
    result = dict()
    for i in xrange(n_files):
        fn = os.path.join(pred_pth, '%05d.npz' % i)
        with np.load(fn) as file_data:
            # Data loaded for each file
            for entry_name in collect_items:
                data_entry = file_data[entry_name]
                if i == 0:
                    result[entry_name] = data_entry
                else:
                    result[entry_name] = np.append(result[entry_name], data_entry, axis=0)
    for entry_name, data_entry in result.items():
        out_fn = out_fns % (pred, entry_name)
        print '  %s: %s (%s)' % (entry_name, str(data_entry.shape), out_fn)
        np.savez(out_fn, data=data_entry)
