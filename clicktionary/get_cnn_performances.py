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

def get_cnn_performance_by_revelation(set_index, source_set_index=None, model_name='VGG16', train_batches_max=16, feature_name='fc7ex', classifier_type='svm'):
    fn = get_cnn_performance_filename(set_index)
    print fn
    if os.path.isfile(fn):
        return pickle.load(open(fn, 'rb'))
    if source_set_index is None: source_set_index = set_index
    pred_fn = util.get_predictions_filename(model_name=model_name, feature_name=feature_name, classifier_type=classifier_type, train_batches=range(0, train_batches_max), set_index=source_set_index, set_name='clicktionary')
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
    # get_cnn_performance_by_revelation(100, 100)
    get_cnn_performance_by_revelation(126, 120)
    # get_cnn_performance_by_revelation(71, 70)
    #get_cnn_performance_by_revelation(122, 120, 'clickme_gradient_VGG16', 20)
    get_cnn_performance_by_revelation(123, 120, 'tf_VGG16', 16)
    get_cnn_performance_by_revelation(125, 120, 'tf_VGG16', 16, 'relu7')
    get_cnn_performance_by_revelation(127, 120, 'VGG16_control', 16, 'fc7ex')
    get_cnn_performance_by_revelation(131, 120, 'VGG16_control', 6, 'fc7ex', 'logreg')
    get_cnn_performance_by_revelation(132, 120, 'VGG16_ft_clickme', 6, 'fc7ex', 'logreg')
    get_cnn_performance_by_revelation(133, 120, 'VGG16_control', 6, 'fc7ex', 'svm')
    get_cnn_performance_by_revelation(134, 120, 'VGG16_ft_clickme', 6, 'fc7ex', 'svm')
    #get_cnn_performance_by_revelation(130)
    #get_cnn_performance_by_revelation(140)

