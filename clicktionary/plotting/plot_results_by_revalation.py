#!/usr/bin/env python
# Plot the image scores by revalation

import os
from rapid_categorization.evaluation.data_loader import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rapid_categorization.clicktionary import config
import pickle
from scipy.stats import norm

def get_cnn_results_by_revalation(set_index, off=0.0, include_true_labels=False):
    pickle_name=os.path.join(config.pickle_path, 'perf_by_revalation_clicktionary_%d.p' % set_index)
    with open(pickle_name) as f:
        data = pickle.load(f)
    if include_true_labels:
        return np.vstack((
            data['revalation_raw'] + off, data['correctness_raw'], data['true_labels'])).transpose()
    else:
        return np.vstack((
                data['revalation_raw'] + off, data['correctness_raw'])).transpose()

def get_human_results_by_revaluation(experiment_run, filename_filter=None, off=0.0):
    data = Data()
    data.load(experiment_run=experiment_run)
    revs, scores = data.get_summary_by_revalation(filename_filter=filename_filter)
    data = np.array([(100.0 - r + off, s) for r, score in zip(revs, scores) for s in score])
    return data

def do_plot(data, clr, label):
    df = pd.DataFrame(data, columns=['Revalation', 'correctness'])
    df = df[df['Revalation'] > 0]
    estimator = np.mean
    ax = sns.regplot(
        data=df, x='Revalation', y='correctness', ci=95, n_boot=10,
        x_estimator=estimator, color=clr, truncate=True, fit_reg=True, order=3, label=label) # , logistic=False
    ax.set_xticks(np.linspace(0, 100, 11))

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

def plot_results_by_revaluation_by_class(set_index, set_name='clicktionary'):
    data_cnn = get_cnn_results_by_revalation(set_index)
    exps = [set(), set()]
    for i ,fn in enumerate(['classes_exp_1.txt', 'classes_exp_2.txt']):
        fn_full = os.path.join('/media/data_cifs/clicktionary/causal_experiment', fn)
        lines = open(fn_full, 'rt').read().splitlines()
        for l in lines:
            classname, cat = l.split(' ')
            exps[int(cat)].add(classname)
    data_human1 = get_human_results_by_revaluation(set_index, set_name, filename_filter=exps[0], off=1)
    data_human2 = get_human_results_by_revaluation(set_index, set_name, filename_filter=exps[1], off=2)

    #data = np.vstack((data_cnn, data_human))
    #humanness = np.vstack((np.zeros((data_cnn.shape[0], 1)), np.ones((data_human.shape[0], 1))))
    #data = np.hstack((data, humanness))
    sns.set_style('white')
    do_plot(data_cnn, 'black', 'CNN')
    do_plot(data_human1, 'red', 'Human Non-animal')
    do_plot(data_human2, 'green', 'Human Animal')
    plt.title('Accuracy by image revalation')
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_by_class_%s_%d_.png' % (set_name, set_index)))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_by_class_%s_%d_.pdf' % (set_name, set_index)))

def plot_human_dprime(experiment_run):
    set_index, set_name = config.get_experiment_sets(experiment_run)
    data_cnn = get_cnn_results_by_revalation(set_index, include_true_labels=True)
    exps = [set(), set()]
    for i ,fn in enumerate(['classes_exp_1.txt', 'classes_exp_2.txt']):
        fn_full = os.path.join('/media/data_cifs/clicktionary/causal_experiment', fn)
        lines = open(fn_full, 'rt').read().splitlines()
        for l in lines:
            classname, cat = l.split(' ')
            exps[int(cat)].add(classname)
    data_animal = get_human_results_by_revaluation(experiment_run, filename_filter=exps[0], off=0)
    data_nonanimal = get_human_results_by_revaluation(experiment_run, filename_filter=exps[1], off=0)
    is_animal = np.vstack((np.ones((data_animal.shape[0], 1)), np.zeros((data_nonanimal.shape[0], 1))))
    data_all = np.hstack((np.vstack((data_animal, data_nonanimal)), is_animal))
    do_plot_dprime(data_all, 'b', 'Human')
    do_plot_dprime(data_cnn, 'r', 'CNN')
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'dprime_by_revalation_%s_%d_.pdf' % (set_name, set_index)))
    plt.savefig(os.path.join(config.plot_path, 'dprime_by_revalation_%s_%d_.png' % (set_name, set_index)))

def plot_results_by_revaluation(experiment_run='clicktionary'):
    set_index, set_name = config.get_experiment_sets(experiment_run)
    data_cnn = get_cnn_results_by_revalation(set_index)
    data_human = get_human_results_by_revaluation(experiment_run, off=1)
    sns.set_style('white')
    do_plot(data_cnn, 'black', 'CNN')
    do_plot(data_human, 'red', 'Human')
    plt.title('Accuracy by image revalation')
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_%s_%d_.png' % (set_name, set_index)))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_%s_%d_.pdf' % (set_name, set_index)))


if __name__ == '__main__':
    experiment_run = 'clicktionary50msfull'
    plot_results_by_revaluation(experiment_run=experiment_run)
    #plt.figure()
    #plot_human_dprime(experiment_run)
    #plot_results_by_revaluation_by_class(set_index=50, set_name='clicktionary')
    #plot_human_dprime(set_index=50)
    plt.show()
