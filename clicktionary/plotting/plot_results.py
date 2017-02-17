import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def create_plot(
        pickle_name='perf_by_revalation_clicktionary_50.p'):
    with open(pickle_name) as f:
        data = pickle.load(f)
    df = pd.DataFrame(
        np.vstack((
            data['revalation_raw'], data['correctness_raw'])).transpose(),
            columns=['Revalation', 'correctness'])
    sns.set_style('white')
    ax = sns.regplot(
        data=df, x='Revalation', y='correctness', ci=95, n_boot=1000,
        x_estimator=np.mean, logistic=True, color='black', truncate=True)
    ax.set_xticks(np.linspace(0, 100, 11))
    plt.savefig('perf_by_revalation_clicktionary_50.pdf')

if __name__ == '__main__':
    create_plot()

