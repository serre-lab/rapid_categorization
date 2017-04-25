#!/usr/bin/env python
#  Collects the list of images for clicktionary
# Stores into a file with lines [filename][TAB][class]
# to be used by the NSF pipeline

import os, re
from rapid_categorization.clicktionary.config import imageset_base_path
from rapid_categorization.clicktionary.imageset_from_raw import convert_raw_imageset

def load_classes():
    fn = os.path.join(imageset_base_path, 'classes.txt')
    class_map = {}
    for line in open(fn, 'rt').read().splitlines():
        ll = line.split(' ')
        class_map[ll[0]] = int(ll[1])
    return class_map

def collect_images(subpath, set_index):
    out_fn = os.path.join(imageset_base_path, 'raw_images_%d.txt' % set_index)
    if os.path.isfile(out_fn):
        print 'Output file exists. skipping.'
        print out_fn
        return
    class_map = load_classes()
    with open(out_fn, 'wt') as fid:
        for root, dirs, files in os.walk(os.path.join(imageset_base_path, subpath)):
            print 'Processing images in %s...' % root
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() != '.png':
                    print 'Skipping %s' % file
                    continue
                class_name = re.findall('[a-zA-Z_]*', file)[0]
                class_index = class_map[class_name]
                full_fn = os.path.join(root, file)
                print class_index, full_fn
                fid.write('%s\t%d\n' % (full_fn, class_index))
    print 'File list written to %s' % out_fn
    convert_raw_imageset(set_index, 'clicktionary')

if __name__ == '__main__':
    #collect_images(subpath='clicktionary_log_scale_masked_images', set_index=80)
    #collect_images(subpath='clicktionary_log_scale_masked_lrp', set_index=90)
    #collect_images(subpath='clicktionary_probabilistic_region_growth', set_index=100)
    #collect_images(subpath='clicktionary_uniform_region_growth', set_index=110)
    collect_images(subpath='clicktionary_probabilistic_region_growth_centered', set_index=120)
    collect_images(subpath='lrp_probabilistic_region_growth_centered', set_index=130)
    collect_images(subpath='fixation_prediction_probabilistic_region_growth_centered', set_index=140)