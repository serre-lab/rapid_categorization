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

def get_cnn_results_by_revalation(set_index, off=0.0):
    pickle_name=os.path.join(config.pickle_path, 'perf_by_revalation_clicktionary_60.p')
    with open(pickle_name) as f:
        data = pickle.load(f)
    return np.vstack((
            data['revalation_raw'] + off, data['correctness_raw'])).transpose()

def get_human_results_by_revaluation(set_index, set_name='clicktionary', filename_filter=None, off=0.0):
    data = Data()
    data.load(set_index=set_index, set_name=set_name)
    revs, scores = data.get_summary_by_revalation(filename_filter=filename_filter)
    data = np.array([(100.0 - r + off, s) for r, score in zip(revs, scores) for s in score])
    return data

def do_plot(data, clr, label):
    df = pd.DataFrame(data, columns=['Revalation', 'correctness'])
    df = df[df['Revalation'] > 0]
    ax = sns.regplot(
        data=df, x='Revalation', y='correctness', ci=95, n_boot=10,
        x_estimator=np.mean, color=clr, truncate=True, fit_reg=True, logistic=True, label=label) # , logistic=False
    ax.set_xticks(np.linspace(0, 100, 11))

def plot_results_by_revaluation_by_class(set_index, set_name='clicktionary'):
    data_cnn = get_cnn_results_by_revalation(set_index)
    exps = [set(), set()]
    for i ,fn in enumerate(['classes_exp_1.txt', 'classes_exp_2.txt']):
        fn_full = os.path.join('/media/data_cifs/clicktionary/causal_experiment/clicktionary_masked_images', fn)
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

def plot_results_by_revaluation(set_index, set_name='clicktionary'):
    data_cnn = get_cnn_results_by_revalation(set_index)
    data_human = get_human_results_by_revaluation(set_index, set_name, off=1)
    sns.set_style('white')
    do_plot(data_cnn, 'black', 'CNN')
    do_plot(data_human, 'red', 'Human')
    plt.title('Accuracy by image revalation')
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_%s_%d_.png' % (set_name, set_index)))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revalation_%s_%d_.pdf' % (set_name, set_index)))


if __name__ == '__main__':
    plot_results_by_revaluation(set_index=50, set_name='clicktionary')
    plt.figure()
    plot_results_by_revaluation_by_class(set_index=50, set_name='clicktionary')
    plt.show()
