#!/usr/bin/env python
#  Collects the list of images for clicktionary
# Stores into a file with lines [filename][TAB][class]
# to be used by the NSF pipeline

import os, re
from rapid_categorization.clicktionary.config import imageset_base_path

def load_classes():
    fn = os.path.join(imageset_base_path, 'classes.txt')
    class_map = {}
    for line in open(fn, 'rt').read().splitlines():
        ll = line.split(' ')
        class_map[ll[0]] = int(ll[1])
    return class_map

def collect_images():
    out_fn = os.path.join(imageset_base_path, 'raw_images_50.txt')
    class_map = load_classes()
    with open(out_fn, 'wt') as fid:
        for root, dirs, files in os.walk(imageset_base_path):
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

if __name__ == '__main__':
    print collect_images()