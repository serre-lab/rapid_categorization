#!/usr/bin/env python

# Load rank data of classifier output and generate set of outliers between two ranks

import numpy as np
import os
import pickle

#set_names = ['gist_svm_setb50k_0-9', 'gist_sgd_svm_setb50k_0-19']
#set_names = ['gist_sgd_svm_setb50k_0-19', 'caffe_fc7_sgd_svm_setb50k_0-19']
set_names = ['hmax_c2rbf_svm_setb50k_0-9', 'caffe_fc7_sgd_svm_setb50k_0-9']

rank_pth = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/predictions'
test_range = range(100000,300000)
n_test = len(test_range)
debug=True

with open(os.path.join(rank_pth, 'source.pickle'), 'rb') as fid:
    source_data = pickle.load(fid)
true_labels = source_data['true_labels'][test_range]

image_ranks = np.zeros((n_test, 2))
for i,set_name in enumerate(set_names):
    rank_fn = os.path.join(rank_pth, set_name + '_ranks.npz')
    with np.load(rank_fn) as rank_file:
        image_ranks[:, i] = rank_file['image_ranks']

corr = np.corrcoef(image_ranks.transpose())[1,0]
if debug:
    import matplotlib.pyplot as plt
    image_ranks_a = image_ranks[true_labels==0,:]
    image_ranks_b = image_ranks[true_labels==1,:]
    plot_batch_size = 100
    plot_num = 10000
    for i in xrange(plot_num / plot_batch_size):
        rng=xrange(i*plot_batch_size, (i+1)*plot_batch_size)
        plt.plot(image_ranks_a[rng, 0], image_ranks_a[rng, 1], 'k.')
        plt.plot(image_ranks_b[rng, 0], image_ranks_b[rng, 1], 'r.')
    plt.title('Prediction ranks (10000 images)\ncorr=%.3f' % corr)
    plt.xlabel(set_names[0])
    plt.ylabel(set_names[1])
    plt.savefig(rank_pth + '/pred_ranks_%s_%s' % (set_names[0], set_names[1]))
    plt.show()