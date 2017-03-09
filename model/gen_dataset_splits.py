#!/usr/bin/env python

# Generate randomized training/validation/test splits

import os
import random
import copy
import numpy as np
from collections import defaultdict

base_path = '/users/seberhar/data/data/AnNonAn_cc256'
#base_path = '/home/sven2/DeleteMe/convtest'

# Read input dataset
all = open(os.path.join(base_path, 'all_sorted.txt'), 'rt').readlines()
random.shuffle(all)
n_all = len(all)
print '%d images found' % n_all

# Create shuffled copy of input array and save to file given by relative path into base_path
def save_set(arr, fn):
    full_fn = os.path.join(base_path, fn)
    if os.path.isfile(full_fn):
        #raise RuntimeError('File %s already exists - aborting!' % full_fn)
        print 'File %s already exists - skipping.' % full_fn
    else:
        with open(full_fn, 'wt') as fid:
            fid.writelines(arr)

# Write shuffled main set
save_set(all, 'all.txt')

# Create balanced subsets
set_size = 5 * 1000
set_name = 'b50k'
# Sort and count entries by label
labels = np.array([int(l.split('\t')[1].replace('\n','')) for l in all])
unique_labels = np.unique(labels)
print 'Found unique labels: %s' % str(unique_labels)
n_categories = unique_labels.size
if set_size % n_categories > 0:
    raise RuntimeError('Error creating balanced sets: Set size (%d) not divisible by category count (%d)!' % (set_size, n_categories))
all_by_label = defaultdict(list)
n_smallest_class = None
for i in xrange(n_all):
    all_by_label[labels[i]] += [all[i]]
for label in unique_labels:
    n = len(all_by_label[label])
    print 'Class %d has %d instances.' % (label, n)
    if (n_smallest_class is None) or (n_smallest_class > n):
        n_smallest_class = n
n_balanced = n_smallest_class * n_categories
print 'Smallest class has %d instances. Creating sets for %d images.' % (n_smallest_class, n_balanced)
# Create sets
n_sets = n_balanced / set_size
n_remainder = n_balanced - n_sets * set_size
set_size_per_class = set_size / n_categories
print "Creating %d balanced sets of %d images each plus one remainder set of %d images." % (n_sets, set_size, n_remainder)
for i in xrange(n_sets):
    set_fn = 'set%s_%05d.txt' % (set_name, i)
    subset = []
    for label in unique_labels:
        subset += all_by_label[label][i * set_size_per_class : (i+1) * set_size_per_class]
    random.shuffle(subset)
    save_set(subset, set_fn)

if n_remainder > 0:
    set_fn = 'set%s_%05d.txt' % (set_name, n_sets)
    subset = []
    for label in unique_labels:
        subset += all_by_label[label][n_sets * set_size_per_class : n_sets * set_size_per_class + n_remainder / n_categories]
    random.shuffle(subset)
    save_set(subset, set_fn)

# Create remainder set of unbalanced images
if n_balanced < n_all:
    set_fn = 'remainder_%s.txt' % (set_name)
    subset = []
    for label in unique_labels:
        n = len(all_by_label[label]) - n_balanced/n_categories
        if n > 0:
            subset += all_by_label[label][-n:]
    random.shuffle(subset)
    save_set(subset, set_fn)


# Create unbalanced subsets (plus one remainder set)
set_size = 5 * 1000
set_name = 'u50k'

n_sets = n_all / set_size
n_remainder = n_all - n_sets * set_size
print "Creating %d unbalanced sets of %d images each plus one remainder set of %d images." % (n_sets, set_size, n_remainder)
for i in xrange(n_sets):
    set_fn = 'set%s_%05d.txt' % (set_name, i)
    subset = all[i * set_size : (i+1) * set_size]
    save_set(subset, set_fn)

if n_remainder > 0:
    set_fn = 'set%s_%05d.txt' % (set_name, n_sets)
    subset = all[-n_remainder:]
    save_set(subset, set_fn)
