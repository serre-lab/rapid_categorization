#!/usr/bin/python
# DEPRECATED. Was used to convert IMRI images
# Take an image list and create .webm videos and the image list in .csv for psiTurk


import csv
import fnmatch
import os
import subprocess
from collections import defaultdict
import Image

# Source and destination files
in_fn = '/home/sven2/psiturk/imriexp/data/experiment_1.csv'
stim_dir = '/home/sven2/psiturk/imriexp/stimuli/experiment_1_stimuli'
out_dir = '/home/sven2/psiturk/bmtcat/static/dataset'
stim_sh = '/home/sven2/psiturk/video/stim2video.sh'
img_list_fn = '/home/sven2/psiturk/bmtcat/static/img_list2.csv'
fix_ms_vals = [1000, 1100, 1200, 1300, 1400, 1500, 1600]
after_time_s = '500'
gen_webm = False
gen_list = True

# Load image list
all_stims = defaultdict(int)
stim_cats = dict()
cnt = 0
with open(in_fn, 'rb') as exp1:
    exp1reader = csv.reader(exp1, delimiter=',')
    is_first = True
    for row in exp1reader:
        if is_first:
            is_first = False
            continue
        fn = row[1]
        cat = row[2].split('_')[0]
        all_stims[fn] += 1
        stim_cats[fn] = cat
        cnt += 1

print '%d tests.' % cnt

# Check if the images are actually there
for root, dirnames, filenames in os.walk(stim_dir):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        relpath_fn = os.path.join(os.path.relpath(root, stim_dir), filename)
        print '%d %s' % (all_stims[relpath_fn], relpath_fn)
        cnt -= all_stims[relpath_fn]

print '%d not found.' % cnt

# Generate .webm videos for all stimuli and all different pre-presentation times
if gen_webm:
    for fn, cnt in all_stims.iteritems():
        base_fn = os.path.splitext(fn)[0]
        out_subdir = os.path.join(out_dir, base_fn)
        os.makedirs(out_subdir)
        jpeg_fn = os.path.join(stim_dir, fn)
        png_fn = os.path.join(out_subdir, 'stim.png')
        Image.open(jpeg_fn).save(png_fn)
        for fix_ms in fix_ms_vals:
            fix_ms_s = str(fix_ms)
            out_fn = os.path.join(out_subdir, fix_ms_s + '.webm')
            print [stim_sh, os.path.join(stim_dir, fn), out_fn, fix_ms_s, after_time_s]
            subprocess.call([stim_sh, png_fn, out_fn, fix_ms_s, after_time_s])

# Generate file list in .csv format to be read by psiTurk javascript
if gen_list:
    f = open(img_list_fn, 'w')
    f.write('img,cat\n')
    for fn, cat in stim_cats.iteritems():
        base_fn = os.path.splitext(fn)[0]
        print '%s\t%s' % (base_fn, cat)
        f.write(base_fn + ',' + cat + '\n')
