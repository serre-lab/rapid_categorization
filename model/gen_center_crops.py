#!/usr/bin/env python

import os
import shutil
import glob
import scipy.ndimage as nd
import PIL.Image

# Iterate over all images in folders and subfolders and generate grayscale, center-cropped versions
root_in_pth = '/users/seberhar/data/data/AnNonAn/IMAGES'
root_out_pth = '/users/seberhar/data/data/AnNonAn_cc256'
categories = ['NON-ANIMAL', 'ANIMAL']
target_size = (256, 256)

# Recursive iteration over files/folders
def do_conv(in_pth, out_pth):
    files = glob.glob(os.path.join(in_pth, '*'))
    if len(files) > 0:
        print '%d items in %s moved to %s' % (len(files), in_pth, out_pth)
        if not os.path.exists(out_pth):
            os.makedirs(out_pth)
        for in_fn in files:
            out_fn = os.path.join(out_pth, os.path.basename(in_fn))
            if os.path.isdir(in_fn):
                do_conv(in_fn, out_fn)
            else:
                # Is this an image file?
                filename, file_extension = os.path.splitext(in_fn)
                if not (file_extension.lower() in ['.png', '.jpeg', '.jpg', '.gif']):
                    print '  Unknown extension. Image skipped: %s' % in_fn
                else:
                    # Conversion always to jpeg
                    out_fn = out_fn.replace(file_extension, '.jpg')
                    # Already converted?
                    if os.path.isfile(out_fn):
                        print '  Skipping %s.' % out_fn
                    else:
                        try:
                            # OK, convert this!
                            img = PIL.Image.open(in_fn).convert('L')
                            w, h = img.size
                            if w > h:
                                img = img.crop(((w-h)/2, 0, (w-h)/2+h, h))
                            elif h >w:
                                img = img.crop((0, (h-w)/2, w, (h-w)/2+w))
                            img = img.resize(target_size, resample=PIL.Image.BICUBIC)
                            # Save converted image
                            img.save(out_fn)
                            print 'saved to %s.' % out_fn
                        except:
                            print 'Error processing image %s. Skipping.' % in_fn

# Work on given categories
for cat in categories:
    do_conv(os.path.join(root_in_pth, cat), os.path.join(root_out_pth, cat))