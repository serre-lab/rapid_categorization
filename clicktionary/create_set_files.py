#!/usr/bin/env python
# Create the set definition files to be loaded by the psiturk experiment

import os
from hmax.levels.util import get_imageset_filename_raw, get_imageset_filename
import numpy as np
from rapid_categorization.clicktionary.config import imageset_base_path

def load_imagelist(set_index=50, set_name='clicktionary'):
    set_fn = get_imageset_filename_raw(set_index=set_index, set_name=set_name)
    class_images = dict()
    for line in open(set_fn, 'rt').read().splitlines():
        ll = line.split('\t')
        fn, _ = os.path.splitext(ll[0])
        class_index = int(ll[1])
        class_images[os.path.basename(fn)] = class_index
    return class_images

def create_set_files(n_sets, input_set_index, input_set_name, output_set_index, output_set_name):
    class_images = load_imagelist(set_index=input_set_index, set_name=input_set_name)
    all_classes = np.unique(class_images.values())
    n_images = len(class_images)
    revalations = xrange(0, 100, 10)
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
                fid.write('%d/%s\t%d\n' % (rev, image, class_images[image]))

if __name__ == '__main__':
    create_set_files(20, 50, 'clicktionary', 1000, 'clicktionary')
