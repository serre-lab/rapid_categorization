#!/usr/bin/env python

from rapid_categorization.clicktionary.config import imageset_base_path
import numpy as np
from hmax.levels.util import get_predictions_filename
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def get_performances(model_name, feature_name, classifier_type, train_batches, set_index, set_name):
    pred_fn = get_predictions_filename(model_name, feature_name, classifier_type, train_batches, set_index, set_name)
    pred = np.load(pred_fn)
    print pred.keys()
    sfn = pred['source_filenames']
    hyper_dist = pred['hyper_dist']
    pred_labels = (hyper_dist > 0) + 0
    true_labels = pred['true_labels']
    correctness = (pred_labels == true_labels).astype(np.float)
    print 'Performance: %.2f%%' % (100 * float(sum(pred_labels == true_labels)) / len(true_labels))
    revalation = 100.0 - np.array([int(fn.split('/')[-2]) for fn in sfn])
    revs = np.unique(revalation)
    print pred_labels
    print true_labels
    perfs = []
    for rev in revs:
        mask = (revalation == rev)
        tl = true_labels[mask]
        pl = pred_labels[mask]
        perf = (100 * float(sum(pl == tl)) / len(tl))
        print 'Performance %02d: %.2f%%' % (rev, perf)
        perfs += [perf]
    return revs, perfs, revalation, correctness

if __name__ == '__main__':
    model_name = 'VGG16'
    feature_name = 'fc7ex'
    classifier_type = 'svm'
    train_batches = [0, 15]
    set_index = 50
    set_name = 'clicktionary'
    revs, perfs, revalation, correctness = get_performances(model_name, feature_name, classifier_type, train_batches, set_index, set_name)
    #mat_data = np.hstack((revalation.reshape([-1, 1]), correctness.reshape([-1, 1])))
    #print mat_data
    #data = pd.DataFrame(mat_data, columns=['revalation', 'correctness'])
    #sns.tsplot(time='revalation', value='correctness', data=data, ci=95, err_style="boot_traces", n_boot=500)
    #sns.tsplot(mat_data[(1, 0),:])
    plt.plot(revs, perfs)
    plt.title('%s(%s) %s performance on %s_%d' % (model_name, feature_name, classifier_type, set_name, set_index))
    plt.xlabel('Revalation (%)')
    plt.ylabel('Percent correct')
    plt.gca().set_ylim([45, 100])
    plt.savefig(os.path.join(imageset_base_path, 'perf_by_revalation_%s_%d.png' % (set_name, set_index)))
    plt.show()