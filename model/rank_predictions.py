#!/usr/bin/env python

# Get test set indices sorted by decision plane distance based on various classifiers

import numpy as np
import os
import glob

pred_pth = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/predictions'
dist_fns = glob.glob(pred_pth + '/*_hyper_dist.npz')

test_range = range(100000,300000)
n_test = len(test_range)

force_rewrite = False

# Rank from each classification
for dist_fn in dist_fns:
    out_fn = dist_fn.replace('_hyper_dist.npz', '_ranks.npz')
    if not force_rewrite:
        if os.path.isfile(out_fn):
            continue
    print out_fn
    with np.load(dist_fn) as dist_file_data:
        dist = dist_file_data['data']
    dist_subset = dist[test_range]
    sorted_dists = sorted(enumerate(dist_subset), key=lambda x:x[1])
    image_by_rank = [test_range[s[0]] for s in sorted_dists]
    image_offset_by_rank = [s[0] for s in sorted_dists]
    image_ranks = np.zeros(n_test)
    image_ranks[image_offset_by_rank] = range(n_test)


    np.savez(out_fn, image_by_rank=image_by_rank, image_ranks=image_ranks)
