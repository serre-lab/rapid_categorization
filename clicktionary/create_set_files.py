#!/usr/bin/env python
# Create the set definition files to be loaded by the psiturk experiment

import os, re
from hmax.levels.util import get_imageset_filename_raw, get_imageset_filename
import numpy as np

def load_imagelist(set_index, set_name, exclusion_files=None):
    excluded_classes = set()
    if exclusion_files is not None:
        # Load a list of excluded classes
        for exclusion_file in exclusion_files:
            for line in open(exclusion_file, 'rt').read().splitlines():
                excluded_classes.add(line.split(' ')[0])
    set_fn = get_imageset_filename_raw(set_index=set_index, set_name=set_name)
    class_images = dict()
    for line in open(set_fn, 'rt').read().splitlines():
        ll = line.split('\t')
        fn, _ = os.path.splitext(ll[0])
        class_name = re.findall('[a-zA-Z_]*', os.path.basename(fn))[0]
        if class_name in excluded_classes:
            print 'Skipping excluded %s' % fn
            continue
        class_index = int(ll[1])
        class_images[os.path.basename(fn)] = class_index
    print '%d images loaded from set %d.' % (len(class_images), set_index)
    return class_images

def create_set_files(n_sets, input_set_index, input_set_name, output_set_index, output_set_name, exclusion_files=None, include_full=False):
    class_images = load_imagelist(set_index=input_set_index, set_name=input_set_name, exclusion_files=exclusion_files)
    all_classes = np.unique(class_images.values())
    n_images = len(class_images)
    revalations = [str(r) for r in xrange(0, 100, 10)]
    if include_full: revalations += ['full']
    n_revalations = len(revalations)
    image_to_index = {}
    idx = 0
    for c in all_classes:
        image_list = [k for k in class_images.keys() if class_images[k] == c]
        for image in np.random.permutation(image_list):
            image_to_index[image] = idx
            idx += 1
    for i_set in xrange(n_sets):
        set_fn = get_imageset_filename(set_index = output_set_index + i_set, set_name = output_set_name)
        with open(set_fn, 'wt') as fid:
            image_list = np.random.permutation([k for k in class_images.keys()])
            for image in image_list:
                rev = revalations[(image_to_index[image] + i_set) % n_revalations]
                fid.write('%s/%s\t%d\n' % (rev, image, class_images[image]))

if __name__ == '__main__':
    exp2 = '/media/data_cifs/clicktionary/causal_experiment/classes_exp_2.txt'
    exp3 = '/media/data_cifs/clicktionary/causal_experiment/classes_exp_3.txt'
    #create_set_files(20, 50, 'clicktionary', 2000, 'clicktionary', exclusion_files=[exp3])
    #create_set_files(20, 50, 'clicktionary', 3000, 'clicktionary', exclusion_files=[exp2, exp3])
    create_set_files(22, 50, 'clicktionary', 4000, 'clicktionary', include_full=True, exclusion_files=[exp2, exp3])
