#!/usr/bin/env python
# Plot the image scores by revalation
import re
import os
from rapid_categorization.evaluation.data_loader import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rapid_categorization.clicktionary import config
import pickle
from scipy.stats import norm
from rapid_categorization.run_settings.settings import get_settings


def apply_log_scale(data_human, data_cnn):
    # Check if CNN and human are in the same scale. If not, recode the CNN.
    uni_human_rev = np.unique(data_human[:, 0])
    uni_cnn_rev = np.unique(data_cnn[:, 0])
    if not all(uni_human_rev == uni_cnn_rev):
        for hr, cr in zip(uni_human_rev, uni_cnn_rev):
            it_idx = data_cnn[:, 0] == cr
            data_cnn[it_idx, 0] = hr
    return data_cnn


def get_cnn_results_by_revelation(set_index, off=0.0, include_true_labels=False):
    pickle_name=os.path.join(config.pickle_path, 'perf_by_revelation_clicktionary_%d.p' % set_index)
    with open(pickle_name) as f:
        data = pickle.load(f)
    if include_true_labels:
        return np.vstack((
            data['revelation_raw'] + off, data['correctness_raw'], data['true_labels'])).transpose()
    else:
        return np.vstack((
                data['revelation_raw'] + off, data['correctness_raw'])).transpose()


def combine_revs_and_scores(revs, scores, off=0.0, is_inverted_rev=True, is_log_scale=False, full_val=200):
    if is_inverted_rev:
        data = np.array([(100.0 - r + off, s) for r, score in zip(revs, scores) for s in score])
    else:
        data = np.array([(r + off, s) for r, score in zip(revs, scores) for s in score])
    if is_log_scale:
        unique_bins = np.unique(data[:, 0])
        full_image = np.argmin(data[:,0])
        max_realization = np.max(data[:,0])
        log_bins = np.logspace(0, np.log10(max_realization), np.sum(unique_bins != full_image))
        remap_index = np.vstack((unique_bins, np.hstack((full_val, log_bins))))
        data = rescale_x(data, remap_index)
    return data


def rescale_x(x, target_x):
    out_x = np.copy(x)
    for idx in range(target_x.shape[1]):
        out_x[x[:, 0] == target_x[0,idx], 0] = target_x[1,idx]
    return out_x


def get_human_results_by_revaluation(
        experiment_run,
        filename_filter=None,
        off=0.0,
        is_inverted_rev=True,
        log_scale=False,
        exclude_workerids=None):
    data = Data()
    data.load(experiment_run=experiment_run, exclude_workerids=exclude_workerids)
    revs, scores = data.get_summary_by_revelation(filename_filter=filename_filter)
    return combine_revs_and_scores(revs, scores, off=off, is_inverted_rev=is_inverted_rev, is_log_scale=log_scale)


def do_plot(data, clr, label, log_scale=False, estimator=np.mean, full_size=200, max_val=None, fit_line=True, ci=66.6):
    df = pd.DataFrame(data, columns=['Revelation', 'correctness'])
    full_size_df = df[df['Revelation'] == full_size]
    if max_val is not None:
        full_size_df.loc[:, 'Revelation'] = max_val
    else:
        max_val = full_size
    df = df[df['Revelation'] < full_size]
    # df = df[df['Revelation'] > 0]
    ax = sns.regplot(
        data=df, x='Revelation', y='correctness', ci=ci, n_boot=200,
        x_estimator=estimator, color=clr, truncate=True, fit_reg=fit_line,
        order=3, label=label) # , logistic=False
    ax = sns.regplot(
        data=full_size_df, x='Revelation', y='correctness', ci=ci, n_boot=200,
        x_estimator=estimator, color=clr, truncate=True)
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim(0.2, max_val + 2000)
        ax.set_xticks(np.hstack((np.around(np.logspace(0, 2, 11), 2), max_val)))
        ax.set_xticklabels([str(x) for x in np.around(np.logspace(0, 2, 11), 2)] + ['Full'], rotation=60)
        ax.set_xlabel('Log-spaced image realization')
    else:
        ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
    ax.set_ylabel('Categorization accuracy (%)')
    plt.tight_layout()


def do_plot_dprime(data, clr, label):
    revs = data[:,0]
    correctness = data[:, 1]
    animal = data[:, 2]
    urevs = np.unique(revs)
    Z = norm.ppf
    dprimes = []
    tps = []
    fas = []
    ms = []
    for rev in urevs:
        idcs = (revs == rev)
        c = correctness[idcs]
        a = animal[idcs]
        n = len(c)
        tp = 0
        fa = 0
        tp = float(sum(c[a==1.0] == 1.0)) / sum(a == 1.0)
        fa = float(sum(c[a==0.0] == 0.0)) / sum(a == 0.0)
        #for cc, aa in zip(c, a):
        #    if (cc == 1.0 and aa == 1.0): tp += 1
        #    if (cc == 0.0 and aa == 1.0): fa += 1
        #tp = float(tp) / sum(a == 1.0)
        #fa = float(fa) / sum(a == 0.0)
        m = np.mean(c)
        dp = Z(tp) - Z(fa)
        dprimes += [dp]
        tps += [tp]
        fas += [fa]
        ms += [m]
    plt.plot(urevs, dprimes, clr, label=label + ' dPrime')
    plt.plot(urevs, tps, clr + '^', label=' true positive')
    plt.plot(urevs, fas, clr + 'v', label = ' false negative')
    plt.plot(urevs, ms, clr + 'o', label = ' mean accuracy')


def plot_results_by_revaluation_by_class(experiment_run, exclude_workerids=None):
    set_index, set_name = config.get_experiment_sets(experiment_run)
    data_cnn = get_cnn_results_by_revelation(set_index)
    exps = [set(), set()]
    for i ,fn in enumerate(['classes_exp_1.txt', 'classes_exp_2.txt']):
        fn_full = os.path.join('/media/data_cifs/clicktionary/causal_experiment', fn)
        lines = open(fn_full, 'rt').read().splitlines()
        for l in lines:
            classname, cat = l.split(' ')
            exps[int(cat)].add(classname)
    data_human1 = get_human_results_by_revaluation(set_index, set_name, filename_filter=exps[0], off=1, exclude_workerids=exclude_workerids)
    data_human2 = get_human_results_by_revaluation(set_index, set_name, filename_filter=exps[1], off=2, exclude_workerids=exclude_workerids)

    #data = np.vstack((data_cnn, data_human))
    #humanness = np.vstack((np.zeros((data_cnn.shape[0], 1)), np.ones((data_human.shape[0], 1))))
    #data = np.hstack((data, humanness))
    sns.set_style('white')
    do_plot(data_cnn, 'black', 'CNN')
    do_plot(data_human1, 'red', 'Human Non-animal')
    do_plot(data_human2, 'green', 'Human Animal')
    plt.title('Accuracy by image revelation')
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_by_class_%s.png' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_by_class_%s.pdf' % experiment_run))


def plot_human_dprime(experiment_run, exclude_workerids=None):
    set_index, set_name = config.get_experiment_sets(experiment_run)
    data_cnn = get_cnn_results_by_revelation(set_index, include_true_labels=True)
    exps = [set(), set()]
    for i ,fn in enumerate(['classes_exp_1.txt', 'classes_exp_2.txt']):
        fn_full = os.path.join('/media/data_cifs/clicktionary/causal_experiment', fn)
        lines = open(fn_full, 'rt').read().splitlines()
        for l in lines:
            classname, cat = l.split(' ')
            exps[int(cat)].add(classname)
    data_animal = get_human_results_by_revaluation(experiment_run, filename_filter=exps[0], off=0, exclude_workerids=exclude_workerids)
    data_nonanimal = get_human_results_by_revaluation(experiment_run, filename_filter=exps[1], off=0, exclude_workerids=exclude_workerids)
    is_animal = np.vstack((np.ones((data_animal.shape[0], 1)), np.zeros((data_nonanimal.shape[0], 1))))
    data_all = np.hstack((np.vstack((data_animal, data_nonanimal)), is_animal))
    do_plot_dprime(data_all, 'r', 'Human')
    do_plot_dprime(data_cnn, 'k', 'CNN')
    plt.title('dPrime by image revelation\n' + config.get_experiment_desc(experiment_run))
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'dprime_by_revelation_%s.pdf' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'dprime_by_revelation_%s.png' % experiment_run))


def plot_results_by_revelation(
        experiment_run='clicktionary',
        exclude_workerids=None,
        fit_line=True):
    sns.set_style('white')
    if isinstance(experiment_run, list):
        # colors = sns.color_palette('Set1', len(experiment_run))
        colors = sns.color_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"][:len(experiment_run)])
        for exp, color in zip(experiment_run, colors):
            p = get_settings(exp)
            set_index, set_name = p['set_index'], p['set_name']
            data_human = get_human_results_by_revaluation(
                exp, off=0, is_inverted_rev=False,
                log_scale=p['log_scale_revelations'],
                exclude_workerids=exclude_workerids)
            exp_params = [x for x in re.split('[A-Za-z]+',exp) if len(x) > 0]
            title = 'Human image time: %s | response time: %s' % (exp_params[0], exp_params[1])
            do_plot(data_human, color, title, log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)
        plt.title('Accuracy by log-spaced image feature revelation\n')
        experiment_run = '_'.join(experiment_run)
    else:
        p = get_settings(experiment_run)
        set_index, set_name = p['set_index'], p['set_name']
        data_human = get_human_results_by_revaluation(
            experiment_run, off=0, is_inverted_rev=False,
            log_scale=p['log_scale_revelations'],
            exclude_workerids=exclude_workerids)
        plt.title('Accuracy by image revelation\n' + p['desc'])
        do_plot(data_human, 'red', 'Human', log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)

    # CNN is always the same
    data_cnn = get_cnn_results_by_revelation(set_index)
    if p['log_scale_revelations']:
        data_cnn = apply_log_scale(data_human, data_cnn)
    do_plot(data_cnn, 'black', 'CNN', log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_%s.png' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_%s.pdf' % experiment_run))
    print 'Saved to: %s' % os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run)


def plot_results_by_revelation_and_max_answer_time(
        experiment_run='clicktionary',
        exclude_workerids=None,
        fit_line=True):
    # Get exp data
    p = get_settings(experiment_run)
    set_index, set_name = p['set_index'], p['set_name']
    # Plot CNN results
    data_cnn = get_cnn_results_by_revelation(set_index)
    sns.set_style('white')
    do_plot(data_cnn, 'black', 'CNN', fit_line=fit_line)
    # Plot human results
    data_human_all = Data()
    data_human_all.load(experiment_run=experiment_run)
    mats = sorted(data_human_all.max_answer_times)
    colors = sns.cubehelix_palette(len(mats))
    for max_answer_time, color in zip(mats, colors):
        revs, scores = data_human_all.get_summary_by_revelation_and_max_answer_time(max_answer_time=max_answer_time)
        data_human = combine_revs_and_scores(revs, scores)
        do_plot(data_human, color, 'Human %dms' % max_answer_time, fit_line=fit_line)
    plt.title('Accuracy by image revelation and max answer time (n=%d)\n' % data_human_all.n_subjects + config.get_experiment_desc(experiment_run))
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.pdf' % experiment_run))
    print 'Saved to: %s' % os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run)


def plot_cnn_comparison(set_indexes):
    colors = ['red', 'black', 'blue']
    for i, set_index in enumerate(set_indexes):
        data_cnn = get_cnn_results_by_revelation(set_index)
        print data_cnn
        do_plot(data_cnn, colors[i], 'CNN %d' % set_index)
    plt.legend()
    outfn = os.path.join(config.plot_path, 'cnn_perf_by_revelation_%s.png' % str(set_indexes))
    plt.savefig(outfn)
    print 'Saved to: %s' % outfn


if __name__ == '__main__':
    # chosen_exp = ['clicklog400ms500msfull']
    #chosen_exp = [['clicklog400ms500msfull', 'clicklog400ms150msfull']]
    #for exp in chosen_exp:
    #    plt.figure()
    #    plot_results_by_revelation(experiment_run=exp)
        #plt.figure()
        #plot_human_dprime(exp)
    #plot_results_by_revaluation_by_class(set_index=50, set_name='clicktionary')
    #plot_human_dprime(set_index=50)
    #plot_cnn_comparison([110, 100])
    plot_results_by_revelation(
        experiment_run='click_probfill',
        exclude_workerids=['A25YG9M911WA3T'],  # In cases like when participants inexplicably are able to complete the experiment twice
        fit_line=False)
    plt.show()
    plt.show()
