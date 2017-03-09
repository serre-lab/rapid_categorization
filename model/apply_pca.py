#!/usr/bin/env python

# Apply dimensionality reduction to feature data

import sys
import os
import pickle
import numpy as np
import rapid_categorization.model.util
from sklearn.decomposition import PCA

if __name__ == "__main__":
    if len(sys.argv) < 2:
        feature_name = 'hmax_c2'

    base_path = rapid_categorization.model.util.get_base_path()
    train_batches = range(0, 10)
    apply_batches = range(0, 200)
    set_name = 'setb50k'
    out_suffix = '_pca10'
    n_components = 2048
    pca_fn = os.path.join(base_path, feature_name, 'pca%d_%s_%d-%d.pickle' % (n_components, set_name, train_batches[0], train_batches[-1]))
    with open(pca_fn, 'rb') as fid:
        pca = pickle.load(fid)

    for i_batch,train_batch in enumerate(apply_batches):
        in_fn = os.path.join(base_path, feature_name, '%s_%05d.npz' % (set_name, train_batch))
        out_fn = os.path.join(base_path, feature_name, '%s_%05d%s.npz' % (set_name, train_batch, out_suffix))
        print 'Loading feature file %s.' % in_fn
        with np.load(in_fn) as file_contents:
            data = file_contents['data']
        print 'Applying PCA...'
        pdata = pca.transform(data)
        print 'Saving to output %s.' % out_fn
        np.savez(out_fn, data=pdata)
