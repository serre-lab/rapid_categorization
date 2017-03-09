# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:03:15 2016

@author: root
"""

#!/usr/bin/env python

# Train incremental linear SVM via SGD on animal data

import sys
import os
import pickle
import numpy as np
import util
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def train_sgd(clf, loss, tr_data, tr_true_labels):
    if clf is None:
        clf = SGDClassifier(loss=loss, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
                            verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,
                            class_weight=None, warm_start=False)
    clf.partial_fit(tr_data,tr_true_labels,classes=[0,1])
    return clf

def train_svm(clf, tr_data, tr_true_labels, props):
    clf = LinearSVC(dual=False, C=10**props['C'], verbose=1)
    clf.fit(tr_data, tr_true_labels)
    return clf

def train_classifier(feature_name, train_batch_num, base_npz_dir, test_batches):
    test_acc = []
    base_path = util.get_base_path()
    categories = util.get_categories()
    train_batches = range(0, train_batch_num)
    #test_batches = range(train_batch_num,train_batch_num+1) JC edit
    set_name = 'setb50k'
    label_set_name = set_name
    subset = ''#'_pca1'
    classifier_paramstring = ''
    if do_norm: classifier_paramstring += 'N'
    if props['C'] != 0:
        classifier_paramstring += 'C%d' % props['C']
    out_fn = os.path.join(base_npz_dir, feature_name, '%s%s_%s%s_%d-%d.pickle' % (classifier_type, classifier_paramstring, set_name, subset, train_batches[0], train_batches[-1]))
    if do_norm:
        out_fn_norm = os.path.join(base_npz_dir, feature_name, 'norm_%s%s_%d.pickle' % (set_name, subset, train_batches[0]))
    print 'Training %s...' % out_fn

    if classifier_type == 'sgd_svm':
        is_incremental = True
    else:
        is_incremental = False

    norm = dict()
    clf = None

    for i_batch,train_batch in enumerate(train_batches + test_batches):
        fn = os.path.join(base_npz_dir, feature_name, '%s_%05d%s.npz' % (set_name, train_batch, subset))
        print 'Processing feature file %s.' % fn
        print fn
        with np.load(fn) as file_contents:
           
            data = file_contents['data']

        true_labels, _ = util.load_labels(label_set_name, train_batch)

        if do_norm:
            if i_batch == 0:
                # Initial batch to determine mean and variance for normalization
                norm['mean'] = np.expand_dims(data.mean(axis=0), 0)
                norm['std'] = np.expand_dims(data.std(axis=0), 0)
                norm['std'] = np.maximum(norm['std'], 0.01)
                with open(out_fn_norm, 'wb') as fid:
                    pickle.dump(norm, fid)

            data -= norm['mean']
            data /= norm['std']
            print 'Data after normalization: Mean %f, Std %f' % (data.mean(axis=0).mean(axis=0), data.std(axis=0).mean(axis=0))


        if is_incremental:
            # Incremental: Do training every training iteration
            # Do testing not just on test but also during training before feeding the new training data
            do_train = (i_batch < len(train_batches))
            do_test = (i_batch > 0)
            use_data = data
            use_true_labels = true_labels
        else:
            # Non-incremental: Train once when all training batches have been collected
            do_train = (i_batch == len(train_batches) - 1)
            do_test = (i_batch >= len(train_batches))
            # data collection phase
            if not do_test:
                if i_batch == 0:
                    data_all = data
                    all_true_labels = true_labels
                else:
                    data_all = np.concatenate((data_all, data), axis=0)
                    all_true_labels = np.concatenate((all_true_labels, true_labels), axis=0)
            use_data = data_all
            use_true_labels = all_true_labels

        print '  use data %s.' % str(use_data.shape)
        print '  use labels %s' % str(use_true_labels.shape)

        if do_test:
            # After some batch training has been done, predict performance
            pred_labels = clf.predict(data)
            acc = float(sum(pred_labels == true_labels)) / true_labels.size
            test_acc.append(acc)
            print '  Batch accuracy: %.1f%%' % (acc * 100)

        if do_train:
            if classifier_type == 'sgd_svm':
                clf = train_sgd(clf, 'hinge', use_data, use_true_labels)
            elif classifier_type == 'svm':
                clf = train_svm(clf, use_data, use_true_labels, props)
            pred_labels = clf.predict(use_data)
            acc = float(sum(pred_labels == use_true_labels)) / use_true_labels.size
            print '  Train accuracy: %.1f%%' % (acc * 100)
            # Dump classifier data at every iteration
            with open(out_fn, 'wb') as fid:
                pickle.dump(clf, fid)
    return np.mean(test_acc)

