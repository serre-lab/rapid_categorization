#!/usr/bin/env python

# Reduce dimensionality of feature data

import sys
import os
import pickle
import numpy as np
import rapid_categorization.levels.util
from sklearn.decomposition import PCA

if __name__ == "__main__":
    if len(sys.argv) < 2:
        feature_name = 'hmaxmultiscale_c2rbf'
    else:
        feature_name = sys.argv[1]
    if len(sys.argv) < 3:
        train_batch_count = 2
    else:
        train_batch_count = int(sys.argv[2])

    base_path = rapid_categorization.levels.util.get_base_path()
    categories = rapid_categorization.levels.util.get_categories()
    train_batches = range(0, train_batch_count)
    set_name = 'setb50k'
    n_components = 2048
    out_fn = os.path.join(base_path, feature_name, 'pca%d_%s_%d-%d.pickle' % (n_components, set_name, train_batches[0], train_batches[-1]))

    for i_batch,train_batch in enumerate(train_batches):
        fn = os.path.join(base_path, feature_name, '%s_%05d.npz' % (set_name, train_batch))
        print 'Loading feature file %s.' % fn
        with np.load(fn) as file_contents:
            if i_batch == 0:
                data = file_contents['data']
            else:
                data = np.concatenate((data, file_contents['data']), axis=0)

    print 'Fitting PCA...'
    pca = PCA(n_components=n_components, copy=False, whiten=False)
    pca.fit(data)
    print 'Done. %s' % out_fn

    # Dump fitted PCA
    with open(out_fn, 'wb') as fid:
        pickle.dump(pca, fid)