#!/usr/bin/env python

# Collect examples where model confidence changes drastically between two sets of feature descriptors

import os
import numpy as np
import util
from predict import collect_predictions
import shutil

def collect_extreme_examples(save_set_index, groups, skip_extremes, n_extremes, classifier_type, train_batches, test_batches, set_name, do_norm):
    # groups: Truple of two lists of (model_name, feature_name) pairs to be averaged and then compared
    group_ranks = []
    mean_preds_correct = []
    for group in groups:
        hyper_dists = None
        for element in group:
            model_name, feature_name = element
            pred_label, true_labels, hyper_dist, source_filenames = collect_predictions(model_name, feature_name, classifier_type, train_batches, test_batches, set_name, do_norm)
            if hyper_dists is None:
                hyper_dists = hyper_dist[:,None]
                pred_labels = pred_label[:,None]
            else:
                hyper_dists = np.concatenate((hyper_dists, hyper_dist[:, None]), axis=1)
                pred_labels = np.concatenate((pred_labels, pred_label[:, None]), axis=1)
        n_samples = len(true_labels)
        mean_dist = hyper_dists.mean(axis=1)
        mean_pred = np.round(pred_labels.mean(axis=1))
        mean_preds_correct += [mean_pred == true_labels]
        mean_dist = mean_dist * (true_labels*2-1)
        sort_index = np.argsort(-mean_dist)
        group_rank = np.zeros(n_samples)
        group_rank[sort_index] = range(n_samples)
        group_ranks += [group_rank]
    # Diff between the averages over the two groups
    correct_change = mean_preds_correct[1].astype(np.int) - mean_preds_correct[0].astype(np.int) # 1 if changed from incorrec tto correct when going from gorup 0 to 1; -1 if reverse
    rank_improvement = group_ranks[0] - group_ranks[1] # Decrease of rank index [i.e. improvement] from group 0 to group 1
    diff = correct_change * 100000 + rank_improvement
    print sorted(diff)
    # Evaluate separated by true label
    c_ext_filenames = []
    c_ext_true_labels = []
    for eval_true_label in [0, 1]:
        # Make subsets
        subset = (true_labels == eval_true_label)
        c_diff = diff[subset]
        c_source_filenames = source_filenames[subset]
        c_true_labels = true_labels[subset]
        # Find extremes
        sort_index = np.argsort(-c_diff) # Find largest diffs
        ext_index = sort_index[skip_extremes/2:n_extremes/2]
        c_ext_filenames += [c_source_filenames[ext_index]]
        c_ext_true_labels += [c_true_labels[ext_index]]
        #print c_diff[ext_index]
        for i_group in [0, 1]:
            preds_correct = mean_preds_correct[i_group][subset][ext_index]
            pc = float(sum(preds_correct)) / ((n_extremes-skip_extremes)/2)
            print('Group %d label %d correct: %.2f' % (i_group, eval_true_label, pc*100))
    ext_filenames = np.concatenate(c_ext_filenames)
    ext_true_labels = np.concatenate(c_ext_true_labels)
    # Save as set
    set_data = ['%s\t%d' % e for e in zip(ext_filenames, ext_true_labels)]
    save_filename = util.get_imageset_filename(save_set_index, set_name)
    with open(save_filename, 'wt') as fid:
        fid.write('\n'.join(set_data))
    # Save as raw filenames
    raw_root = util.get_input_image_root(save_set_index, set_name)
    set_data_raw = [os.path.join(raw_root, e + '.jpg') for e in ext_filenames]
    save_filename_raw = util.get_imageset_filename_raw(save_set_index, set_name)
    with open(save_filename_raw, 'wt') as fid:
        fid.write('\n'.join(set_data_raw))
    # Create imageset folders
    output_folder = save_filename.replace('.txt', '')
    if os.path.isdir(output_folder):
        os.rmdir(output_folder)
    os.makedirs(output_folder)
    for fn in ext_filenames:
        jpgfn = fn+'.jpg'
        shutil.copyfile(os.path.join(util.get_input_image_root(save_set_index, set_name), jpgfn), os.path.join(output_folder, jpgfn))
    # Create montage
    os.system('montage $( head -n 15 "%s"; head -n %d "%s" | tail -n 15 ) %s.jpg' % (save_filename_raw, (n_extremes-skip_extremes)/2+15, save_filename_raw, save_filename_raw))
    print save_filename_raw

if __name__ == '__main__':
    #low = [('VGG16', layer+'ex') for layer in ('conv1_1', 'conv1_2', 'conv2_1')]
    #mid = [('VGG16', layer+'ex') for layer in ('conv4_3', 'conv5_1', 'conv5_2')]
    #high = [('VGG16', layer + 'ex') for layer in ('conv5_3', 'fc6', 'fc7')]
    low = [('VGG16', layer+'ex') for layer in ('conv1_2',)]
    mid = [('VGG16', layer+'ex') for layer in ('conv5_2',)]
    high = [('VGG16', layer + 'ex') for layer in ('fc7',)]
    #collect_extreme_examples(80011, (low, mid), 0, 200, 'svm', range(16), range(16, 20), 'set', True)
    #collect_extreme_examples(80012, (mid, low), 0, 200, 'svm', range(16), range(16, 20), 'set', True)
    #collect_extreme_examples(80013, (mid, high), 0, 200, 'svm', range(16), range(16, 20), 'set', True)
    #collect_extreme_examples(80014, (high, mid), 0, 200, 'svm', range(16), range(16, 20), 'set', True)
    collect_extreme_examples(80033, (mid, high), 0, 220, 'svm', range(16), range(16, 20), 'set', True)
    collect_extreme_examples(80034, (high, mid), 0, 220, 'svm', range(16), range(16, 20), 'set', True)
