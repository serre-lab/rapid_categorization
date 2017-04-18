#!/usr/bin/env python2
# Extract CNN performances sorted by revelation

import os, pickle
import numpy as np
from rapid_categorization.clicktionary.config import pickle_path
from rapid_categorization.model import util

def get_cnn_performance_filename(set_index):
    return os.path.join(pickle_path, 'perf_by_revelation_clicktionary_%d.p' % set_index)

def filename_to_revelation(fn):
    s = fn.split('/')[0]
    if s == 'full': return 200
    return int(s)

def get_cnn_performance_by_revelation(set_index, source_set_index=None):
    fn = get_cnn_performance_filename(set_index)
    print fn
    if os.path.isfile(fn):
        return pickle.load(open(fn, 'rb'))
    if source_set_index is None: source_set_index = set_index
    pred_fn = util.get_predictions_filename(model_name='VGG16', feature_name='fc7ex', classifier_type='svm', train_batches=range(0, 16), set_index=source_set_index, set_name='clicktionary')
    data = np.load(open(pred_fn, 'rb'))
    print data.keys()
    revelations = np.array([filename_to_revelation(fns) for fns in data['source_filenames']])
    out_data = {}
    out_data['revelation_raw'] = revelations
    out_data['correctness_raw'] = data['true_labels'] == data['pred_labels']
    out_data['true_labels'] = data['true_labels']
    out_data['source_filenames'] = data['source_filenames']
    return pickle.dump(out_data, open(fn, 'wb'))
    return out_data

if __name__ == '__main__':
    get_cnn_performance_by_revelation(100, 100)
    get_cnn_performance_by_revelation(110, 110)
    get_cnn_performance_by_revelation(71, 70)
