#!/usr/bin/env python
# Plot the image scores by revalation
import re
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import clicktionary_plots
from rapid_categorization.clicktionary import config
from rapid_categorization.evaluation.data_loader import Data
from rapid_categorization.run_settings.settings import get_settings
import matplotlib
from sortedcontainers import SortedDict
from tqdm import tqdm


matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Helvetica')
matplotlib.rc('text', usetex='false')
matplotlib.rcParams.update({'font.size': 12})


def apply_log_scale(data_human, data_cnn, off=0):
    # Check if CNN and human are in the same scale. If not, recode the CNN.
    uni_human_rev = np.unique(data_human[:, 0])
    uni_cnn_rev = np.unique(data_cnn[:, 0])
    if not all(uni_human_rev == uni_cnn_rev):
        for hr, cr in zip(uni_human_rev, uni_cnn_rev):
            it_idx = data_cnn[:, 0] == cr
            data_cnn[it_idx, 0] = hr + off
    return data_cnn


def get_cnn_results_by_revelation(
        set_index,
        off=0.0,
        include_true_labels=False,
        filter_class_file=None):
    pickle_name = os.path.join(
        config.pickle_path, 'perf_by_revelation_clicktionary_%d.p' % set_index)
    with open(pickle_name) as f:
        data = pickle.load(f)
    if filter_class_file is not None:
        with open(
            os.path.join(
                config.imageset_base_path, filter_class_file)) as f:
            classes = f.readlines()
        classes = [x.split(' ')[0] for x in classes]
        rev_raw = np.asarray([])
        correct_raw = np.asarray([])
        true_labs = np.asarray([])
        for idx in range(len(data['source_filenames'])):
            if any([x in data['source_filenames'][idx] for x in classes]):
                rev_raw = np.append(rev_raw, data['revelation_raw'][idx])
                correct_raw = np.append(
                    correct_raw, data['correctness_raw'][idx])
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


def combine_revs_and_scores(
        revs,
        scores,
        subject_list,
        off=0.0,
        is_inverted_rev=True,
        is_log_scale=False,
        full_val=200
        ):
    if is_inverted_rev:
        data = np.array([(
            100.0 - r + off, s, su) for r, score, subjects in zip(
            revs, scores, subject_list) for s, su in zip(
            score, subjects)])
    else:
        data = np.array([(
            r + off, s, su) for r, score, subjects in zip(
            revs, scores, subject_list) for s, su in zip(
            score, subjects)])
    if is_log_scale:
        unique_bins = np.unique(data[:, 0])
        full_image = np.argmin(data[:, 0])
        max_realization = np.max(data[:, 0])
        log_bins = np.logspace(
            0, np.log10(max_realization), np.sum(unique_bins != full_image))
        remap_index = np.vstack((unique_bins, np.hstack((full_val, log_bins))))
        data = rescale_x(data, remap_index)
    return data


def rescale_x(x, target_x):
    out_x = np.copy(x)
    for idx in range(target_x.shape[1]):
        out_x[x[:, 0] == target_x[0, idx], 0] = target_x[1, idx]
    return out_x


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def get_human_results_by_revaluation(
        experiment_run,
        filename_filter=None,
        off=0.0,
        is_inverted_rev=True,
        log_scale=False,
        exclude_workerids=None,
        data_filter='full_median',
        return_ims=True
        ):
    data = Data()
    data.load(
        experiment_run=experiment_run,
        exclude_workerids=exclude_workerids
        )
    if 'full_median' in data_filter:
        # Select the people to throw out based on full image acc
        subject_filter = [k for k, v in data.full_accuracies.iteritems() if v > 0.5]
        print 'Removing %s subjects with accuracy less than 50%% on full images.' % (
            len(data.full_accuracies.keys()) - len(subject_filter))
    elif 'response' in data_filter:
        # Do it based on repeated button presses (Linsley et al 2014 method)
        var_data = np.asarray([
                np.var(x) for x in data.response_log.values()]).reshape(-1, 1)
        subject_filter = np.where(var_data > (
            np.median(var_data) - (4 * mad(var_data))))[0]
        print 'Removing %s low-variance subjects.' % (
            len(data.response_log.keys()) - len(subject_filter))
    elif 'timeouts' in data_filter:
        to_data = data.sub_timeouts.values()
        threshold = np.mean(to_data) + (3 * np.std(to_data))
        subject_filter = [k for k, v in data.sub_timeouts.iteritems() if v < threshold]
        print 'Removing %s low-variance subjects.' % (
            len(data.response_log.keys()) - len(subject_filter))
    elif 'zero' in data_filter:
        # Select the people to throw out based on full image acc
        subject_filter = [k for k, v in data.full_accuracies.iteritems() if v > (2. / 6.)]
        print 'Removing %s subjects with accuracy of 0%% on full images.' % (
            len(data.full_accuracies.keys()) - len(subject_filter))
    else:
        subject_filter = [k for k in data.full_accuracies.keys()]
    revs, scores, subject_list, ims = data.get_summary_by_revelation(
        filename_filter=filename_filter,
        subject_filter=subject_filter)
    revs_scores = combine_revs_and_scores(
        revs=revs,
        scores=scores,
        subject_list=subject_list,
        off=off,
        is_inverted_rev=is_inverted_rev,
        is_log_scale=log_scale)
    if return_ims:
        return revs_scores, ims, data.per_subject_rev_acc_mean
    else: 
        return revs_scores


def do_plot(
        data,
        clr,
        label,
        log_scale=False,
        estimator=np.mean,
        full_size=200,
        max_val=None,
        fit_line='linear',
        ci=66.6,
        plot_chance=0.5,
        order=3,
        x_jitter=0,
        plot_type='reg',
        n_boot=500,
        new_max=np.logspace(0, 2.2, num=12)[-1],
        ax=None):

    plot_params = {
        'label': label,
        'x_jitter': x_jitter,
        'x_estimator': estimator,
        'n_boot': n_boot,
        'ci': ci,
        'color': clr,
        'truncate': True,
        }

    if data.shape[1] > 2:
        df = pd.DataFrame(
            data[:, :3],
            columns=['Revelation', 'correctness', 'subject'])  # , 'images'])
        plot_params['units'] = 'subject'
    else:
        df = pd.DataFrame(
            data[:, :2],
            columns=['Revelation', 'correctness'])  # , 'images'])
    # df = df.set_index('Revelation')['correctness']
    # sns.tsplot(data=df)
    # Interpret fit from fit_line
    if 'linear' in fit_line:
        plot_params['logistic'] = True
        plot_params['fit_reg'] = True
    elif 'logistic' in fit_line:
        plot_params['logx'] = True
    elif 'high_order' in fit_line:
        plot_params['order'] = order
    elif 'lowess' in fit_line:
        plot_params['lowess'] = True
        plot_params['fit_reg'] = True
    else:
        plot_params['logistic'] = False
        plot_params['fit_reg'] = False
    if full_size in data:
        df.loc[df.Revelation == full_size, ['Revelation']] = new_max
    else:
        new_max = df.Revelation.max()

    sns.set_style('ticks')
    sns.axes_style({'font.family': ['sans-serif'],
                'font.sans-serif': ['Helvetica']
                })

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    sns.regplot(ax=ax, data=df, x='Revelation', y='correctness', **plot_params)
    sns.despine()
    ax.set_ylim([0.4, 1])
    if log_scale:
        # ax.set_xscale('log')
        # ax.set_xlim(0.2, new_max + 2000)
        xticks = np.around(
            np.logspace(0, 2, 11), 2)
        ax.set_xticks(np.hstack((
            xticks, new_max)))
        xticklabs = [str(x) if y > 5 or x == 1 else '' for x, y in zip(xticks, np.hstack((0, xticks)))] 
        ax.set_xticklabels(xticklabs + ['Full'], rotation=60)
        ax.set_xlabel('Percentage of image revealed', fontweight='bold')
        ax.set_xlim([-4, new_max + 4])
    else:
        ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])
    ax.set_ylabel('Categorization accuracy (%)', fontweight='bold')
    plt.tight_layout()
    if plot_chance is not None:
        plt.axhline(plot_chance, color='gray', linestyle='dashed', linewidth=2)
    mean_values = {x: df[df['Revelation'] == x]['correctness'].mean()
            for x in np.unique(df['Revelation'])}
    return mean_values, ax


def plot_cnn_comparison(set_indexes, labels=None, colors=['red', 'black', 'blue']):
    if labels is None:
        labels = ['CNN %d' % set_index for set_index in set_indexes]
    ax = None
    for i, (set_index, lab) in enumerate(zip(set_indexes, labels)):
        data_cnn = get_cnn_results_by_revelation(set_index, filter_class_file='classes_exp_1.txt')
        _, ax = do_plot(data_cnn, colors[i], lab, ax=ax)
    plt.legend()
    outfn = os.path.join(config.plot_path, 'cnn_perf_by_revelation_%s.png' % str(set_indexes))
    plt.savefig(outfn)
    print 'Saved to: %s' % outfn


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
        human_ci=66.6,
        cnn_ci=66.6,
        human_fit_line=[''],  # ['linear','logistic'],
        cnn_fit_line=[''],  # ['linear','logistic'],
        human_colorpallete='Set2',
        cnn_colorpallete='Greys',
        fit_line='linear',
        data_filter=''):
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
    colors = sns.color_palette(human_colorpallete, len(exps))
    data_human = [get_human_results_by_revaluation(
        experiment_run=experiment_run,
        filename_filter=x,
        off=0,
        is_inverted_rev=is_inverted_rev,
        log_scale=p['log_scale_revelations'],
        exclude_workerids=exclude_workerids,
        data_filter=data_filter,
        return_ims=False)
        for idx, x in enumerate(exps)]
    ax = None
    for x, c, l in zip(data_human, colors, exps):
        _, ax = do_plot(
            x,
            c,
            list(l),
            log_scale=p['log_scale_revelations'],
            ci=human_ci,
            fit_line=human_fit_line,
            ax=ax)

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
    colors = sns.color_palette(cnn_colorpallete, len(exps))[0]
    _, ax = do_plot(
        data_cnn,
        colors,
        'CNN',
        log_scale=p['log_scale_revelations'],
        ci=cnn_ci,
        fit_line=cnn_fit_line,
        ax=ax)
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


def repackage_per_subject(data):
    # Expect per subject keys, with revelations nested inside that
    output = {k: [] for k in data[data.keys()[0]].keys()}
    for v in data.values():
        for k, w in v.iteritems():
            output[k] += [w]
    return output

def plot_results_by_revelation(
        experiment_run='clicktionary',
        exclude_workerids=None,
        human_fit_line=['linear'],
        cnn_fit_line=['linear'],
        human_color_pallete='Set1',
        cnn_colorpallete='Greys',
        human_labels=None,
        cnn_labels='VGG16 performance on clicktionary maps',
        cnn_index=None,
        human_ci=95,
        cnn_ci=95,
        data_filter='',
        max_val=200):
    human_dfs = {}
    human_ims = {}
    human_stats = {}
    cnn_dfs = {}
    ax = None
    cnn_means = None
    if isinstance(experiment_run, list):
        # colors = sns.color_palette('Set1', len(experiment_run))
        colors = sns.color_palette(human_color_pallete, len(experiment_run))
        if len(human_fit_line) < len(experiment_run):
            human_fit_line = np.repeat(human_fit_line, len(experiment_run))
            print 'Expanding fit_lines to match size of experiment_run'
        for idx, (exp, color, fl) in enumerate(
                zip(experiment_run, colors, human_fit_line)):
            p = get_settings(exp)
            set_index = p['set_index']
            data_human, ims, stats = get_human_results_by_revaluation(
                exp,
                off=0,
                is_inverted_rev=False,
                log_scale=p['log_scale_revelations'],
                exclude_workerids=exclude_workerids,
                data_filter=data_filter
                )
            if human_labels is None:
                exp_params = [x for x in re.split(
                    '[A-Za-z]+', exp) if len(x) > 0]
                title = 'Human image time: %s | response time: %s' % (
                    exp_params[0], exp_params[1])
            else:
                title = human_labels[idx]
            human_dfs[exp] = data_human
            human_ims[exp] = ims
            human_stats[exp] = stats
            _, ax = do_plot(
                data_human,
                color,
                title,
                log_scale=p['log_scale_revelations'],
                max_val=max_val,
                fit_line=fl,
                ci=human_ci,
                ax=ax)

        plt.title('Accuracy by log-spaced image feature revelation\n')
        experiment_run = '_'.join(experiment_run)
    else:
        p = get_settings(experiment_run)
        set_index = p['set_index'],
        data_human, ims, stats = get_human_results_by_revaluation(
            experiment_run,
            off=0,
            is_inverted_rev=False,
            log_scale=p['log_scale_revelations'],
            exclude_workerids=exclude_workerids,
            data_filter=data_filter)
        human_dfs[experiment_run] = data_human
        human_ims[exp] = ims
        human_stats[exp] = stats
        plt.title('Accuracy by image revelation\n' + p['desc'])
        _, ax = do_plot(
            data_human,
            '#fc8d59',
            'Human',
            log_scale=p['log_scale_revelations'],
            max_val=max_val,
            fit_line=human_fit_line,
            ax=ax)

    if len(human_stats) == 2:
        dfs = [pd.DataFrame(x, columns=['Revelation', 'correctness', 'subject']) for x in human_dfs.values()]
        human_dfs = []
        for it_df in dfs:
            human_dfs += [{x: it_df[it_df['Revelation'] == x].groupby('subject').mean()['correctness'].as_matrix() for x in it_df['Revelation'].unique()}]
        # packaged_data = [repackage_per_subject(v) for v in human_dfs]
        p_values = measure_p_value_2_sample(*human_dfs)  # (*packaged_data)
        print 'Disabled human/cnn stats until we get synthetic subjects'
        print p_values

    # CNN is always the same
    if 'cnn_class_file' in p.keys():
        filter_class_file = p['cnn_class_file']
    else:
        filter_class_file = None
    if cnn_index is None:
        cnn_index = set_index
    if isinstance(cnn_index, list):
        colors = sns.color_palette(cnn_colorpallete, len(cnn_index))
        if len(cnn_fit_line) < len(cnn_index):
            cnn_fit_line = np.repeat(cnn_fit_line, len(experiment_run))
            print 'Expanding fit_lines to match size of experiment_run'
        for idx, (si, color, la, fl) in enumerate(zip(
                cnn_index, colors, cnn_labels, cnn_fit_line)):
            data_cnn = get_cnn_results_by_revelation(
                si, filter_class_file=filter_class_file, off=float(idx)/2)
            if p['log_scale_revelations']:
                data_cnn = apply_log_scale(data_human, data_cnn, off=float(idx*2))
            cnn_dfs[cnn_index[0]] = data_cnn
            cnn_means, ax = do_plot(
                data_cnn,
                color,
                la,
                log_scale=p['log_scale_revelations'],
                max_val=max_val,
                fit_line=fl,
                ci=cnn_ci,
                ax=ax)
    else:
        data_cnn = get_cnn_results_by_revelation(
            set_index,
            filter_class_file=filter_class_file)
        if p['log_scale_revelations']:
            data_cnn = apply_log_scale(data_human, data_cnn)
        cnn_dfs['set_index'] = data_cnn
        cnn_means, ax = do_plot(
            data_cnn,
            '#91bfdb',
            cnn_labels,
            log_scale=p['log_scale_revelations'],
            max_val=200,
            fit_line=cnn_fit_line,
            ci=cnn_ci,
            ax=ax)

    if human_stats.keys() and cnn_means is not None:
        p_values = {k: measure_p_value_human_v_cnn(
            v, cnn_means) for k, v in human_stats.iteritems()}
        # print 'Disabled human/cnn stats until we get synthetic subjects'
        # print p_values
        # Count per bin # of people that exceed machines.

    plt.legend(loc='upper left')
    plt.savefig(
        os.path.join(
            config.plot_path, 'perf_by_revelation_%s.png' % experiment_run))
    plt.savefig(
        os.path.join(
            config.plot_path, 'perf_by_revelation_%s.pdf' % experiment_run))
    print 'Saved to: %s' % os.path.join(
        config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run)
    return human_dfs, cnn_dfs, human_ims, None


def measure_p_value_human_v_cnn(human, cnn):
    human_keys = {}
    cnn_keys = sorted([float(x) for x in cnn.keys()])
    for idx, k in enumerate(
        sorted(
            [int(x) for x in human[human.keys()[0]].keys()])):
        if k < 0:
            human_keys[np.max(cnn_keys)] = str(k)  # np.max(cnn_keys)
        else:
            human_keys[cnn_keys[idx-1]] = str(k)  # cnn_keys[idx-1]

    combined_human = {}  # SortedDict()
    p_values = {}  # SortedDict()
    # for k in human[human.keys()[0]].keys():
    for k, v in human_keys.iteritems():
        combined_human[k] = []

        for itv in human.values():
            for j, w in itv.iteritems():
                if v == j or v == str(j):
                    combined_human[k] += [w]
        p_values[k] = (np.sum(
            np.asarray(combined_human[k]) > cnn[k]) + 1) / float(
            len(combined_human[k]) + 1)
    combined_human = SortedDict(combined_human)
    p_values = SortedDict(p_values)


def measure_p_value_2_sample(sample_1, sample_2):
    s1_keys = sample_1.keys()  # sorted([float(x) for x in sample_1.keys()])
    s2_keys = sample_2.keys()  # sorted([float(x) for x in sample_2.keys()])
    p_values = {}  # SortedDict()
    for k, v in tqdm(zip(s1_keys, s2_keys), desc='Running randomization tests'):
        p_values[k] = randomization_test(sample_1[k], sample_2[v])
    return SortedDict(p_values)


def randomization_test(s1, s2):
    # Randomize based on the smaller sample
    ss = np.argmin([len(s1), len(s2)])
    if ss == 0:
        return run_randomizations(s1=s1, s2=s2)
    else:
        return run_randomizations(s1=s2, s2=s1)

def run_randomizations(s1, s2, iterations=int(1e4)):
    # Assuming unpaired
    T = np.mean(s1) - np.mean(s2)
    t = np.zeros(iterations)
    # s1 is smaller
    s1_len = len(s1)
    s2_len = len(s2)
    len_diff = s2_len - s1_len
    for idx in range(iterations):
        s2_to_s1 = np.random.rand(s1_len) < .5
        s1_to_s2 = np.hstack((s2_to_s1, np.zeros(len_diff, dtype=bool)))
        it_s1 = np.copy(s1)
        it_s2 = np.copy(s2)[np.random.permutation(s2_len)]
        to_s2 = it_s1[s2_to_s1]
        to_s1 = it_s2[s1_to_s2]
        it_s2[s1_to_s2] = to_s2
        it_s1[s2_to_s1] = to_s1
        t[idx] = np.mean(it_s1) - np.mean(it_s2)
    return (np.sum(T < t) + 1).astype(float) / np.asarray(iterations + 1).astype(float)

def plot_results_by_revelation_and_max_answer_time(
        experiment_run='clicktionary',
        exclude_workerids=None,
        fit_line=True):
    # Get exp data
    p = get_settings(experiment_run)
    set_index = p['set_index']
    # Plot CNN results
    data_cnn = get_cnn_results_by_revelation(set_index)
    do_plot(data_cnn, 'black', 'CNN', fit_line=fit_line)
    # Plot human results
    data_human_all = Data()
    data_human_all.load(experiment_run=experiment_run)
    mats = sorted(data_human_all.max_answer_times)
    colors = sns.cubehelix_palette(len(mats))
    ax = None
    for max_answer_time, color in zip(mats, colors):
        revs, scores, subjects, ims = data_human_all.get_summary_by_revelation_and_max_answer_time(
            max_answer_time=max_answer_time)
        data_human = combine_revs_and_scores(revs, scores, subjects, ims)
        _, ax = do_plot(data_human, color, 'Human %dms' % max_answer_time, fit_line=fit_line, ax=ax)
    plt.title('Accuracy by image revelation and max answer time (n=%d)\n' % data_human_all.n_subjects + config.get_experiment_desc(experiment_run))
    plt.legend()
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run))
    plt.savefig(os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.pdf' % experiment_run))
    print 'Saved to: %s' % os.path.join(config.plot_path, 'perf_by_revelation_and_mat_%s.png' % experiment_run)


if __name__ == '__main__':
    repeat_workers = ['A15FXHC1CVNW31']  # Remove a specific dude
    data_filter = 'none'  # 'response'  # 'response'  # 'full_median' 'response' 'zero'
    image_dir = '/media/data_cifs/clicktionary/causal_experiment/clicktionary_probabilistic_region_growth_centered'
    output_dir = '/home/drew/Desktop/clicktionary_MIRCs'
    mirc_plot_key = 'click_center_probfill_400stim_150res_combined'
    exp_ci = 66.6
    cnn_ci = 0
    show_figs = True

    # # Figure: Clicktionary vs. VGG16
    # human_data, cnn_data, human_ims, p_values = plot_results_by_revelation(
    #     experiment_run=['click_center_probfill_400stim_150res_combined'],
    #     exclude_workerids=repeat_workers,  # In cases like when participants inexplicably are able to complete the experiment twice
    #     human_fit_line=['logistic'],  # ['linear','logistic'],
    #     cnn_fit_line=[],  # ['linear','logistic'],
    #     cnn_index=[],  # [120, 130, 140],
    #     human_labels=[
    #         'Human performance: Clicktionary centered probabilistic; 400ms stim, 150ms response.',
    #         ],
    #     cnn_labels=[
    #         ],
    #     human_ci=exp_ci,
    #     cnn_ci=cnn_ci,
    #     data_filter=data_filter)
    # if show_figs: plt.show()

    # # Figure: Clicktionary vs. LRP
    # plot_results_by_revelation(
    #     experiment_run=[
    #         'click_center_probfill_400stim_150res_combined',
    #         'lrp_center_probfill_400stim_150res_combined'
    #         ],
    #     exclude_workerids=repeat_workers,
    #     human_fit_line=['logistic', 'logistic'],  # ['linear','logistic'],
    #     cnn_fit_line=[],  # ['linear','logistic'],
    #     cnn_index=[],  # [120, 130, 140],
    #     human_labels=[
    #         'Human performance: Clicktionary centered probabilistic; 400ms stim, 150ms response.',
    #         'Human performance: LRP centered probabilistic; 400ms stim, 150ms response.',
    #         ],
    #     cnn_labels=[
    #         ],
    #     human_ci=exp_ci,
    #     cnn_ci=cnn_ci,
    #     data_filter=data_filter)
    # if show_figs: plt.show()

    # Figure: Clicktionary vs. LRP
    # plot_results_by_revelation(
    #     experiment_run=[
    #         'click_center_probfill_400stim_150res_combined',
    #         'lrp_center_probfill_400stim_150res_combined'
    #         ],
    #     exclude_workerids=repeat_workers,
    #     human_fit_line=['logistic', 'logistic'],  # ['linear','logistic'],
    #     cnn_fit_line=['linear', 'linear'],  # ['linear','logistic'],
    #     cnn_index=[120, 130],  # [120, 130, 140],
    #     human_labels=[
    #         'Human performance: Clicktionary centered probabilistic; 400ms stim, 150ms response.',
    #         'Human performance: LRP centered probabilistic; 400ms stim, 150ms response.',
    #         ],
    #     cnn_labels=[
    #         'VGG16 performance: Clicktionary centered probabilistic.',
    #         'VGG16 performance: LRP centered probabilistic.',
    #         ],
    #     human_ci=exp_ci,
    #     cnn_ci=cnn_ci,
    #     data_filter=data_filter)
    # if show_figs: plt.show()

    # plot_results_by_revelation(
    #     experiment_run=[
    #         'click_center_probfill_400stim_150res_combined',
    #         'lrp_center_probfill_400stim_150res_combined'
    #         ],
    #     exclude_workerids=repeat_workers,
    #     human_fit_line=['logistic', 'logistic'],  # ['linear','logistic'],
    #     cnn_fit_line=['linear', 'linear'],  # ['linear','logistic'],
    #     cnn_index=[120,  130],  # [120, 130, 140],
    #     human_labels=[
    #         'Human performance: Animal vehicle centered probabilistic; 400ms stim, 150ms response.',
    #         'Human performance: LRP centered probabilistic; 400ms stim, 150ms response.',
    #         ],
    #     cnn_labels=[
    #         'VGG16 performance: Clicktionary centered probabilistic.',
    #         'VGG16 performance: LRP centered probabilistic.',
    #         ],
    #     human_ci=exp_ci,
    #     cnn_ci=cnn_ci,
    #     data_filter=data_filter)
    # if show_figs: plt.show()


    plot_results_by_revelation(
        experiment_run=[
            'click_center_probfill_400stim_150res_combined'
            ],
        exclude_workerids=repeat_workers,
        human_fit_line=['logistic'],  # ['linear','logistic'],
        cnn_fit_line=['linear', 'lowess', 'lowess', 'lowess'],  # ['linear','logistic'],
        cnn_index=[120, 121, 122, 123],  # [120, 130, 140],
        human_labels=[
            'Human performance: Clicktionary centered probabilistic; 400ms stim, 150ms response.',
            ],
        cnn_labels=[
            'VGG16 performance: Clicktionary centered probabilistic.',
            'Clickme performance: Clicktionary centered probabilistic.',
            'Clickme performance: Clicktionary centered probabilistic. 100k',
            'Baseline performance: Clicktionary centered probabilistic. 100k',
            ],
        human_ci=exp_ci,
        cnn_ci=cnn_ci,
        data_filter=data_filter)
    if show_figs: plt.show()





    # Figure: Clicktionary animal/non-animal
    plot_results_by_revaluation_by_class(
        experiment_run='click_center_probfill_400stim_150res_combined',
        exclude_workerids=repeat_workers,
        human_fit_line='',  # ['linear','logistic'],
        cnn_fit_line='',  # ['linear','logistic'],
        class_file='classes_exp_1.txt',
        human_ci=exp_ci,
        cnn_ci=cnn_ci,
        data_filter=data_filter,
        fit_line='')  # 'all_classes_exp_1.txt'
    if show_figs: plt.show()

    # Figure: Clicktionary vs. LRP 300ms response
    plot_results_by_revelation(
        experiment_run=[
            'click_center_probfill_400stim_300res',
            'lrp_center_probfill_400stim_300res'
            ],
        exclude_workerids=repeat_workers,
        human_fit_line=['', ''],  # ['linear','logistic'],
        cnn_fit_line=['linear'],  # ['linear','logistic'],
        cnn_index=[120],  # [120, 130, 140],
        human_labels=[
            'Human performance: Clicktionary centered probabilistic; 400ms stim, 300ms response.',
            'Human performance: LRP centered probabilistic; 400ms stim, 300ms response.',
            ],
        cnn_labels=[
            'VGG16 performance: Clicktionary centered probabilistic.',
            ],
        human_ci=exp_ci,
        cnn_ci=cnn_ci,
        data_filter=data_filter)
    if show_figs: plt.show()

    clicktionary_plots.find_MIRCS(
        human_data[mirc_plot_key],
        human_ims[mirc_plot_key],
        image_dir=image_dir,
        output_dir=output_dir
        )
