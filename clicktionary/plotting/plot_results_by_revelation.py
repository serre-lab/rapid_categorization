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


def get_cnn_results_by_revelation(set_index, off=0.0, include_true_labels=False, filter_class_file=None):
    pickle_name=os.path.join(config.pickle_path, 'perf_by_revelation_clicktionary_%d.p' % set_index)
    with open(pickle_name) as f:
        data = pickle.load(f)
    if filter_class_file is not None:
        with open(os.path.join(config.imageset_base_path, filter_class_file)) as f:
            classes = f.readlines()
        classes = [x.split(' ')[0] for x in classes]
        rev_raw = np.asarray([])
        correct_raw = np.asarray([])
        true_labs = np.asarray([])
        for idx in range(len(data['source_filenames'])):
            if any([x in data['source_filenames'][idx] for x in classes]):
                rev_raw = np.append(rev_raw, data['revelation_raw'][idx])
                correct_raw = np.append(correct_raw, data['correctness_raw'][idx])
                true_labs = np.append(true_labs, data['true_labels'][idx])
    else:
        rev_raw = data['revelation_raw']
        correct_raw = data['correctness_raw']
        true_labs = data['true_labels']

    if include_true_labels:
        return np.vstack((
            rev_raw + off, correct_raw, true_labs)).transpose()
    else:
        return np.vstack((
                rev_raw + off, correct_raw)).transpose()


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


def do_plot(
        data,
        clr,
        label,
        log_scale=False,
        estimator=np.mean,
        full_size=200,
        max_val=None,
        fit_line=True,
        ci=66.6,
        plot_chance=0.5,
        order=1):
    df = pd.DataFrame(data, columns=['Revelation', 'correctness'])
    # df = df[df['Revelation'] > 0]
    ax = sns.regplot(
        data=df, x='Revelation', y='correctness', ci=ci, n_boot=200,
        x_estimator=estimator, color=clr, truncate=True, fit_reg=fit_line,
        order=order, label=label) # , logistic=False
    if full_size in data:
        full_size_df = df[df['Revelation'] == full_size]
        if max_val is not None:
            full_size_df.loc[:, 'Revelation'] = max_val
        else:
            max_val = full_size
        df = df[df['Revelation'] < full_size]
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
    if plot_chance is not None:
        plt.axhline(plot_chance, color='gray', linestyle='dashed', linewidth=2)
    mean_values = np.asarray(
        [df[df['Revelation'] == x]['correctness'].mean()
            for x in np.unique(df['Revelation'])])
    return mean_values

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


def plot_results_by_revaluation_by_class(
        experiment_run,
        class_file,
        exclude_workerids=None,
        is_inverted_rev=False,
        ci=66.6,
        colorpallete='Set2',
        fit_line=True):
    p = get_settings(experiment_run)
    set_index = p['set_index']

    # Parse class file
    exps = []
    for i, fn in enumerate([class_file]):
        fn_full = os.path.join(
            '/media/data_cifs/clicktionary/causal_experiment', fn)
        lines = open(fn_full, 'rt').read().splitlines()
        for l in lines:
            classname, cat = l.split(' ')
            cat = int(cat)
            if len(exps) <= cat:
                exps.append(set())
            exps[cat].add(classname)

    # Plot Human
    colors = sns.color_palette(colorpallete, len(exps))
    data_human = [get_human_results_by_revaluation(
        experiment_run=experiment_run,
        filename_filter=x,
        off=0,
        is_inverted_rev=is_inverted_rev,
        log_scale=p['log_scale_revelations'],
        exclude_workerids=exclude_workerids)
        for idx, x in enumerate(exps)]
    [do_plot(
        x,
        c,
        list(l),
        log_scale=p['log_scale_revelations'],
        ci=ci,
        fit_line=fit_line) for x, c, l in zip(
        data_human, colors, exps)]

    # Plot CNN
    if 'cnn_class_file' in p.keys():
        filter_class_file = p['cnn_class_file']
    else:
        filter_class_file = None
    print set_index
    data_cnn = get_cnn_results_by_revelation(
        set_index, filter_class_file=filter_class_file)
    if p['log_scale_revelations']:
        data_cnn = apply_log_scale(data_human[0], data_cnn)
    sns.set_style('white')
    do_plot(data_cnn,
        'black',
        'CNN',
        log_scale=p['log_scale_revelations'],
        ci=ci,
        fit_line=fit_line)
    plt.title('Accuracy by image revelation')
    plt.legend()
    plt.savefig(
        os.path.join(
            config.plot_path,
            'perf_by_revelation_by_class_%s.png' % experiment_run))


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
        fit_line=True,
        colorpallete='Set2',
        human_labels=None,
        cnn_label='VGG16 performance'):
    sns.set_style('white')
    if isinstance(experiment_run, list):
        # colors = sns.color_palette('Set1', len(experiment_run))
        colors = sns.color_palette(colorpallete, len(experiment_run))
        for idx, (exp, color) in enumerate(zip(experiment_run, colors)):
            p = get_settings(exp)
            set_index, set_name = p['set_index'], p['set_name']
            data_human = get_human_results_by_revaluation(
                exp, off=0, is_inverted_rev=False,
                log_scale=p['log_scale_revelations'],
                exclude_workerids=exclude_workerids)
            if human_labels is None:
                exp_params = [x for x in re.split('[A-Za-z]+',exp) if len(x) > 0]
                title = 'Human image time: %s | response time: %s' % (exp_params[0], exp_params[1])
            else:
                title = human_labels[idx]
            do_plot(data_human, color, title, log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)
        plt.title('Accuracy by log-spaced image feature revelation\n')
        experiment_run = '_'.join(experiment_run)
        human_means = []
    else:
        p = get_settings(experiment_run)
        set_index, set_name = p['set_index'], p['set_name']
        data_human = get_human_results_by_revaluation(
            experiment_run, off=0, is_inverted_rev=False,
            log_scale=p['log_scale_revelations'],
            exclude_workerids=exclude_workerids)
        plt.title('Accuracy by image revelation\n' + p['desc'])
        human_means = do_plot(data_human, '#fc8d59', 'Human', log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)

    # CNN is always the same
    if 'cnn_class_file' in p.keys():
        filter_class_file = p['cnn_class_file']
    else:
        filter_class_file = None
    print set_index
    data_cnn = get_cnn_results_by_revelation(set_index, filter_class_file=filter_class_file)
    if p['log_scale_revelations']:
       data_cnn = apply_log_scale(data_human, data_cnn)
    cnn_means = do_plot(data_cnn, '#91bfdb', cnn_label, log_scale=p['log_scale_revelations'], max_val=200, fit_line=fit_line)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_%s.png' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_%s.pdf' % experiment_run))
    print 'Saved to: %s' % os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run)
    return human_means, cnn_means


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


def plot_cnn_comparison(set_indexes, labels=None, colors=['red', 'black', 'blue']):
    if labels is None:
        labels = ['CNN %d' % set_index for set_index in set_indexes]
    sns.set_style('white')
    for i, (set_index, lab) in enumerate(zip(set_indexes, labels)):
        data_cnn = get_cnn_results_by_revelation(set_index, filter_class_file='classes_exp_1.txt')
        do_plot(data_cnn, colors[i], lab)
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
    # plot_cnn_comparison(
    #     [120, 130, 140],
    #     labels=['Human Realization Maps', 'VGG16 LRP', 'VGG16 Salience'],
    #     colors=['#fc8d59', '#2ca25f', '#91bfdb'])
    # plot_cnn_comparison(
    #     [110, 120],
    #     labels=['Uncentered Human Realization Maps', 'Centered Human Realization Maps'],
    #     colors=['#fc8d59', '#91bfdb'])
    # plt.show()
    human_means, cnn_means = plot_results_by_revelation(
        experiment_run=['click_center_probfill_650', 'lrp_center_probfill_650'],
        exclude_workerids=['A25YG9M911WA3T'],  # In cases like when participants inexplicably are able to complete the experiment twice
        fit_line=False,
        human_labels=[
            'Human performance: Clicktionary centered probabilistic; 50ms stim, 650ms response.',
            'Human performance: VGG16 LRP centered probabilistic; 50ms stim, 650ms response.',
            ])
    plt.show()


    # plot_results_by_revaluation_by_class(
    #     experiment_run='click_center_probfill_650',
    #     exclude_workerids=['A25YG9M911WA3T'],
    #     class_file='classes_exp_1.txt',
    #     ci=66.6,
    #     fit_line=False)  # 'all_classes_exp_1.txt'
    # plt.show()
    #TODO find MIRCs by np.argmax(diff( .5 + human_means))
