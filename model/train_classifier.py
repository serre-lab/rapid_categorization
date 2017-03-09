#!/usr/bin/env python

# Train incremental linear SVM via SGD on animal data

import sys
import os
import pickle
import numpy as np
import util
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import util

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

def train_logreg(props,tr_data,tr_true_labels):
    clf = LogisticRegression(penalty='l2',dual=False,C=10**props['C'],verbose=1)
    clf.fit(tr_data,tr_true_labels)
    return clf

def train_classifier(model_name, feature_name, classifier_type, props, set_name, train_batch_num, do_norm, test_batches):
    test_acc = []
    train_batches = range(0, train_batch_num)
    label_set_name = set_name
    out_fn = util.get_classifier_filename(model_name, feature_name, classifier_type, train_batches, set_name=set_name, do_norm=do_norm, C=props['C'])
    if do_norm:
        out_fn_norm = util.get_norm_filename(model_name, feature_name, train_batches[0], set_name)
    print 'Training %s...' % out_fn

    if classifier_type == 'sgd_svm':
        is_incremental = True
    else:
        is_incremental = False

    norm = dict()
    clf = None

    for i_batch,train_batch in enumerate(train_batches + test_batches):
        fn = util.get_feature_filename(model_name, feature_name, train_batch, set_name)
        print 'Processing feature file %s.' % fn
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
            elif classifier_type == 'logreg':
                clf = train_logreg(props,use_data,use_true_labels)
            pred_labels = clf.predict(use_data)
            acc = float(sum(pred_labels == use_true_labels)) / use_true_labels.size
            print '  Train accuracy: %.1f%%' % (acc * 100)
            # Dump classifier data at every iteration
            with open(out_fn, 'wb') as fid:
                pickle.dump(clf, fid)
    return np.mean(test_acc)

if __name__ == "__main__":
    props = dict()
    do_norm = True
    if len(sys.argv) < 2:
        model_name = 'VGG16'
        feature_name = 'fc6ex'
    else:
        model_name, feature_name = sys.argv[1].split('_')
    if len(sys.argv) < 3:
        classifier_type = 'svm'
    else:
        classifier_type = sys.argv[2]
    if len(sys.argv) < 4:
        props['C'] = -3
    else:
        props['C'] = int(sys.argv[3])
    if len(sys.argv) < 5:
        train_batch_num = 16
    else:
        train_batch_num = int(sys.argv[4])
    test_batches = [1603241729]

    train_classifier(model_name, feature_name, classifier_type, props, 'set', train_batch_num, do_norm, test_batches)
