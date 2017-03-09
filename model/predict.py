#!/usr/bin/env python

# Create prediction for dataset based on classifiers

import sys
import os
import pickle
import numpy as np
import util
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def do_predictions(data_fn, source_fn, norm_fn, clf_fn, out_fn, do_norm):
    # Load all data
    with np.load(data_fn) as file_contents:
        data = file_contents['data']
    with open(source_fn, 'rt') as fid:
        source_data = fid.read().splitlines()
    source_filenames = np.array([s.split('\t')[0] for s in source_data])
    true_labels = np.array([int(s.split('\t')[1]) for s in source_data])
    # Load classifier
    if do_norm:
        with open(norm_fn, 'rb') as fid:
            norm = pickle.load(fid)
    with open(clf_fn, 'rb') as fid:
        clf = pickle.load(fid)
    # Normalize
    if do_norm:
        data -= norm['mean']
        data /= norm['std']
    # Do predictions
    hyper_dist = clf.decision_function(data)
    clf.classes_ = np.array([0,1])
    pred_labels = clf.predict(data)
    #logs = clf.predict_log_proba(data)
    acc = float(sum(pred_labels == true_labels)) / true_labels.size
    print 'Accuracy = %.1f' % (acc * 100)
    if out_fn is not None:
        np.savez(out_fn, pred_labels=pred_labels, true_labels=true_labels, hyper_dist=hyper_dist, source_filenames=source_filenames)
    return acc

def collect_predictions(model_name, feature_name, classifier_type, train_batches, test_batches, set_name, do_norm):
    pred_labels = []
    true_labels = []
    hyper_dist = []
    source_filenames = []
    for i in test_batches:
        pred_filename = util.get_predictions_filename(model_name, feature_name, classifier_type, train_batches, i, set_name)
        data = np.load(pred_filename)
        pred_labels = np.concatenate((pred_labels, data['pred_labels']))
        true_labels = np.concatenate((true_labels, data['true_labels']))
        hyper_dist = np.concatenate((hyper_dist, data['hyper_dist']))
        source_filenames = np.concatenate((source_filenames, data['source_filenames']))
    return pred_labels, true_labels, hyper_dist, source_filenames

def get_accuracy_for_sets(model_name, feature_name, set_name, test_batches, norm_fn, clf_fn, do_norm):
    # Load all data
    accs = []
    for i in test_batches:
        data_fn = util.get_feature_filename(model_name, feature_name, i, set_name)
        source_fn = util.get_imageset_filename(i, set_name)
        accs += [do_predictions(data_fn, source_fn, norm_fn, clf_fn, None, do_norm)]
    return np.mean(accs)


#Test
if __name__ == '__main__':
    pred_labels, true_labels, hyper_dist, source_filenames = collect_predictions('VGG16', 'conv5_3ex', 'svm', range(16), range(20), 'set', True)
    print source_filenames[:20]
