#!/usr/bin/env python

from rapid_categorization.clicktionary.config import imageset_base_path
import numpy as np
from hmax.levels.util import get_predictions_filename
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle

def filename_to_revelation(fn, invert_rev):
    rev_part = fn.split('/')[-2]
    if rev_part == 'full': return 110.0;
    if invert_rev:
        return 100.0 - int(rev_part)
    else:
        return int(rev_part)

def logscale_revs(revs):
    return list(np.logspace(0.0, 2.0, len(revs)-1)) + [200.0]

def get_performances(model_name, feature_name, classifier_type, train_batches, set_index, set_name, invert_rev, logscale_rev):
    pred_fn = get_predictions_filename(model_name, feature_name, classifier_type, train_batches, set_index, set_name)
    pred = np.load(pred_fn)
    sfn = pred['source_filenames']
    hyper_dist = pred['hyper_dist']
    pred_labels = (hyper_dist > 0) + 0
    true_labels = pred['true_labels']
    correctness = (pred_labels == true_labels).astype(np.float)
    print 'Performance overall: %.2f%%' % (100 * float(sum(pred_labels == true_labels)) / len(true_labels))
    revelation = np.array([filename_to_revelation(fn, invert_rev) for fn in sfn])
    revs = np.unique(revelation)
    if logscale_rev:
        revs_log = logscale_revs(revs)
        rmap = { 200.0: 200.0 }
        for r, rl in zip(revs, revs_log):
            rmap[r] = rl
        revs = np.array(revs_log)
        revelation = np.array([rmap[r] for r in revelation])
    print np.unique(revelation)
    perfs = []
    for rev in revs:
        mask = (revelation == rev)
        tl = true_labels[mask]
        pl = pred_labels[mask]
        perf = (100 * float(sum(pl == tl)) / len(tl))
        print 'Performance %02d: %.2f%%' % (rev, perf)
        perfs += [perf]
    return revs, perfs, revelation, correctness, true_labels

if __name__ == '__main__':
    model_name = 'VGG16'
    feature_name = 'fc7ex'
    classifier_type = 'svm'
    train_batches = [0, 15]
    set_index = 80
    set_name = 'clicktionary'
    logscale_rev = True
    revs, perfs, revelation, correctness, true_labels = get_performances(model_name, feature_name, classifier_type, train_batches, set_index, set_name, invert_rev=False, logscale_rev=logscale_rev)
    #mat_data = np.hstack((revelation.reshape([-1, 1]), correctness.reshape([-1, 1])))
    #print mat_data
    #data = pd.DataFrame(mat_data, columns=['revelation', 'correctness'])
    #sns.tsplot(time='revelation', value='correctness', data=data, ci=95, err_style="boot_traces", n_boot=500)
    #sns.tsplot(mat_data[(1, 0),:])
    data_fn = os.path.join(imageset_base_path, 'perf_by_revelation_%s_%d.p' % (set_name, set_index))
    pickle.dump({'unique_revs': revs, 'mean_perfs': perfs, 'revelation_raw': revelation, 'correctness_raw': correctness, 'true_labels': true_labels}, open(data_fn, 'wt'))
    if logscale_rev:
        plt.semilogx(revs, perfs)
    else:
        plt.plot(revs, perfs)
    plt.title('%s(%s) %s performance on %s_%d' % (model_name, feature_name, classifier_type, set_name, set_index))
    plt.xlabel('Revelation (%)')
    plt.ylabel('Percent correct')
    plt.gca().set_ylim([45, 100])
    plotfn = os.path.join(imageset_base_path, 'perf_by_revelation_%s_%d.png' % (set_name, set_index))
    plt.savefig(plotfn)
    print plotfn
    plt.show()