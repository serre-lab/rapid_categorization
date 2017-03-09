#!/usr/bin/env python

# List all image files for categories in text format with each line being:
# path/to/filename.ext\tcategory_index
# suitable for caffe input

import glob
import os

#base_path = '/home/sven2/DeleteMe/convtest'
base_path = '/users/seberhar/data/data/AnNonAn_cc256'
out_fn_sorted = os.path.join(base_path, 'all_sorted.txt')
categories = ['NON-ANIMAL', 'ANIMAL']

# Recursive file enumeration
def enum_files(fid, base_path, path, idx):
    files = glob.glob(os.path.join(path, '*'))
    for fn in files:
        if os.path.isdir(fn):
            enum_files(fid, base_path, fn, idx)
        else:
            filename, file_extension = os.path.splitext(fn)
            if file_extension.lower() in ['.png', '.jpeg', '.jpg', '.gif']:
                rel_fn = os.path.relpath(fn, base_path)
                fid.write('%s\t%d\n' % (rel_fn, idx))

# Create "sorted" file list
with open(out_fn_sorted, 'wt') as fid:
    # Work on given categories
    for idx,cat in enumerate(categories):
        enum_files(fid, base_path, os.path.join(base_path, cat), idx)
