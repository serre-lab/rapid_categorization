import re
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def ismember(a, b):
    matches = []
    [matches.append(i == b) for i in a]
    return np.sum(np.asarray(matches), axis=0) == 1


def read_txt(file):
    with open(file) as f:
       data = [l.rstrip() for l in f]
    return data


def re_search(tuples, re_string, index):
    return np.asarray(
        [re.search(re_string, t[index]).group() for t in tuples])


def create_plot(realization, guess, name):
    fig = plt.figure()
    df = pd.DataFrame(
        np.vstack((
            realization, guess)).transpose(),
            columns=['Revelation', 'Accuracy'])
    sns.set_style('white')
    ax = sns.regplot(
        data=df, x='Revelation', y='Accuracy', ci=95, n_boot=1000,
        x_estimator=np.mean, logistic=True, color='black', truncate=True)
    ax.set_xticks(np.linspace(0, 100, 11))
    plt.savefig(name + '.pdf')
    plt.close(fig)

if __name__ == '__main__':
    data_list = 'images_50.txt'
    exp_prefix = 'classes_exp_'
    exp_names = [1, 2, 3]

    data = read_txt(data_list)
    image_tuples = np.asarray(
        [(re.split('\s+', r)[0], re.split('\s+', r)[1]) for r in data])
    realization = re_search(image_tuples, '(?<=\/)(\d+)(?=\/)', 0).astype(int)
    category = re_search(image_tuples, '(?<=\/)[a-zA-Z_]+(?=\d)', 0)
    guess = np.asarray([int(t[1]) for t in image_tuples])

    # Load the pickle
    with open('perf_by_revalation_clicktionary_50.p') as f:
        cnn_data = pickle.load(f)['correctness_raw']
 
    for exp in exp_names:
        exp_cats = read_txt('%s%s.txt' % (exp_prefix, str(exp)))
        proc_cats = [re.split('\s', r)[0] for r in exp_cats]
        match_idx = ismember(proc_cats, category)
        exp_realization = realization[match_idx][::-1]
        # exp_guess= guess[match_idx]
        exp_guess = cnn_data[match_idx][::-1]
        create_plot(exp_realization, exp_guess, '%s%s' % (exp_prefix, str(exp)))







