#!/usr/bin/env python2

import os
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from rapid_categorization.clicktionary.get_cnn_performances import get_cnn_performance_by_revelation
from rapid_categorization.clicktionary.config import plot_path

def create_plot(set_index, source_set_index):
    data = get_cnn_performance_by_revelation(set_index, source_set_index)
    df = pd.DataFrame(
        np.vstack((
            data['revelation_raw'], data['correctness_raw'])).transpose(),
            columns=['Revelation', 'correctness'])
    sns.set_style('white')
    ax = sns.regplot(
        data=df, x='Revelation', y='correctness', ci=95, n_boot=1000,
        x_estimator=np.mean, logistic=True, color='black', truncate=True)
    ax.set_xticks(np.linspace(0, 100, 11))
    fn_out = os.path.join(plot_path, 'perf_by_revelation_clicktionary_%d.pdf' % set_index)
    plt.savefig(fn_out)
    print fn_out

if __name__ == '__main__':
    create_plot(100, 100)
    plt.show()
