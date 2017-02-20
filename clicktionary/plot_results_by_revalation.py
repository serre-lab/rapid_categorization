#!/usr/bin/env python
# Plot the image scores by revalation

import os
from rapid_categorization.evaluation.data_loader import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rapid_categorization.clicktionary import config

def plot_results_by_revaluation(set_index, set_name='clicktionary'):
    data = Data()
    data.load(set_index=set_index, set_name=set_name)
    revs, scores = data.get_summary_by_revalation()
    data = np.array([(100.0 - r, s) for r, score in zip(revs, scores) for s in score])
    df = pd.DataFrame(data, columns=['Revalation', 'correctness'])
    sns.set_style('white')
    ax = sns.regplot(
        data=df, x='Revalation', y='correctness', ci=95, n_boot=1000,
        x_estimator=np.mean, logistic=True, color='black', truncate=True)
    ax.set_xticks(np.linspace(0, 100, 11))
    plt.title('Human accuracy by image revalation')
    plt.savefig(os.path.join(config.plot_path, 'human_perf_by_revalation_%s_%d_.png' % (set_name, set_index)))
    plt.savefig(os.path.join(config.plot_path, 'human_perf_by_revalation_%s_%d_.pdf' % (set_name, set_index)))

if __name__ == '__main__':
    plot_results_by_revaluation(set_index=50, set_name='clicktionary')
    plt.show()
