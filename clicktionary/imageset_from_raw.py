#!/usr/bin/env python
# Convert a raw imageset file (containing full paths) to the reduced set (containing only revalation and filename without extension)

import os
from hmax.levels.util import get_imageset_filename, get_imageset_filename_raw

def filename_to_fileid(fn):
    return os.path.splitext('/'.join(fn.split('/')[-2:]))[0]

def convert_raw_to_fileid(raw_filename, fileid_filename):
    data = open(raw_filename, 'rt').read().splitlines()
    with open(fileid_filename, 'wt') as fid:
        for line in data:
            ll = line.split('\t')
            fid.write('%s\t%s\n' % (filename_to_fileid(ll[0]), ll[1]))

def convert_raw_imageset(set_index, set_name):
    fileid_filename = get_imageset_filename(set_index=set_index, set_name=set_name)
    raw_filename = get_imageset_filename_raw(set_index=set_index, set_name=set_name)
    if os.path.isfile(fileid_filename):
        raise RuntimeError('File exists: %s' % fileid_filename)
    convert_raw_to_fileid(raw_filename, fileid_filename)
    print 'Written to %s' % fileid_filename


if __name__ == '__main__':
    convert_raw_imageset(70, 'clicktionary')