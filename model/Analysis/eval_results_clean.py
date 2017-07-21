#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:14:15 2016

@author: jcader
"""

# For remote plotting without display
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pickle
from scipy import stats
from rapid_categorization.model import util
import data_loader
import plot_correlation_overview
import plot_correlations
import matplotlib.pyplot as plt
from sdt import dprime_mAFC
from plot_easy_hard_mod import easy_hard_mod

def eval_human_behaviour(data):
    # Human accuracy
    if len(data.human_acc_bootstrapped):
        hum_accs = data.human_acc_bootstrapped
        hum_acc = np.median(hum_accs)
        hum_acc_p = np.percentile(hum_accs, [2.5, 97.5])
        hum_acc_pm = (hum_acc_p[1] - hum_acc_p[0]) / 2
        hum_acc_d = dprime_mAFC(hum_acc, 2)
        hum_acc_d1 = dprime_mAFC(hum_acc_p[0], 2)
        hum_acc_d2 = dprime_mAFC(hum_acc_p[1], 2)
        hum_acc_d_pm = (hum_acc_d2 - hum_acc_d1) / 2
        print 'Overall human percent correct: %.1f pm %.1f' % (hum_acc*100, hum_acc_pm*100)
        print 'Overall human percent correct d prime: %.3f pm %.3f' % (hum_acc_d, hum_acc_d_pm)
    print '%d trials, %d timeouts (%.1f%%)' % (data.n_trials, data.n_timeouts, float(data.n_timeouts) / data.n_trials * 100)
    print 'Mean reaction time on valid trials per image: %dms pm %dms' % (np.mean(data.rt_data), np.std(data.rt_data))
    print 'Mean correct reaction time: %dms pm %dms' % (np.mean(data.correct_rts), np.std(data.correct_rts))
    # Human agreement
    haa = data.human_acc_by_image_bootstrapped
    if len(haa):
        corrs = []
        iters = 1000
        for i in xrange(iters):
            ivals = np.random.choice(len(haa), 2, replace=False)
            #corr = np.corrcoef(haa[ivals[0]], haa[ivals[1]])[1][0]
            corr = stats.spearmanr(haa[ivals[0]], haa[ivals[1]])[0]
            corrs += [corr]
        corr_vals = np.percentile(corrs, [2.5, 50, 97.5])
        corr_dev = (corr_vals[2] - corr_vals[0]) / 2
        print 'Human correlation between bootstraps: %.2f pm %.2f' % (corr_vals[1], corr_dev)
        print corrs


def eval_correlation_experiment():
    #input lists
    do_eval = True
    save_eval = False
    do_overview = True
    do_easy_hard_plot = False
    do_correlations = False
    do_bootstrap = False
    do_model_eval = True
    model_names_compare = ['VGG16_ft70000', 'VGG16', 'VGG19', 'AlexNet']
    model_names_specific = ['VGG16_ft70000']
    corrs = ["Spearman's rho","Pearson's r", "Kendall's tau"]
    adjust_corr_for_true_label = True
    if adjust_corr_for_true_label:
        data_filename = 'last_data.p'
    else:
        data_filename = 'last_data_unadjusted.p'

    corr = corrs[0]
    experiment_ids = [30, 31, 32, 33, 34]
    classifier_type = 'svm'
    train_batches = range(16)
    axis_types = ['rf']#['rf', 'idx']
    bootstrap_count = 300
    bootstrap_size = 180

    # Data evaluation
    if do_eval:
        data = data_loader.Data()
        data.load_multi(experiment_ids)
        data.eval_participants()
        if do_model_eval:
            for model_name in model_names_compare:
                data.load_model_data(model_name, classifier_type, train_batches)
                data.calc_model_correlation(model_name, corr, adjust_corr_for_true_label)
        # Bootstrapping on correlations
        if do_bootstrap:
            data.bootstrap(experiment_ids, model_names_compare, classifier_type, train_batches, corr_type=corr, adjust_corr_for_true_label=adjust_corr_for_true_label, bootstrap_count=bootstrap_count, bootstrap_size=bootstrap_size)
        if save_eval: pickle.dump(data, open(data_filename, 'wb'))
    else:
        data = pickle.load(open(data_filename, 'rb'))

    # Info: Human accuracy
    #eval_human_behaviour(data)

    # Plots
    if do_overview:
        plot_corr_errs = [m in model_names_specific for m in model_names_compare]
        #plot_correlation_overview.overviewPlot(data, model_names_compare, axis_type='ridx', use_bootstrap_value=do_bootstrap, plot_corr_errs=plot_corr_errs, use_subplots=True)
        plot_correlation_overview.overviewPlot(data, model_names_compare, axis_type='ridx',
                                               use_bootstrap_value=do_bootstrap, plot_corr_errs=plot_corr_errs,
                                               use_subplots=True, is_unadjusted=not adjust_corr_for_true_label)
        plot_correlation_overview.overviewPlot(data, model_names_specific, axis_type='names', use_bootstrap_value=do_bootstrap, plot_corr_errs=do_bootstrap, use_subplots=False, is_unadjusted=not adjust_corr_for_true_label)
    if do_correlations:
        for model_name in model_names_specific:
            layer_names = util.get_model_layers(model_name)
            for layer in layer_names:
                plot_correlations.plot_corr_correct(data, model=model_name, layer=layer)
    if do_easy_hard_plot:
        data.load_im2path(experiment_ids)
        easy_hard_mod(data, model_names_specific[0], 'fc7ex')


def eval_extreme_experiment():
    #input lists
    do_eval = True
    save_eval = True
    experiment_ids = [80020]
    class_ids = [80033, 80034]
    class_names = ['high good, mid bad', 'mid good, high bad']
    do_bootstrap = True
    bootstrap_count = 10
    bootstrap_size = 250

    # Data evaluation
    if do_eval:
        data = data_loader.Data()
        data.load_multi(experiment_ids)
        data.eval_participants()
        data.eval_by_classes(class_idcs=class_ids, class_names=class_names)
        if save_eval: pickle.dump(data, open('extreme_data.p', 'wb'))
    else:
        data = pickle.load(open('extreme_data.p', 'rb'))



if __name__ == '__main__':
    eval_correlation_experiment()
    #eval_extreme_experiment()

# Show and wait for closing
plt.show()
