#!/usr/bin/env python
# oops

import os
import numpy as np
import PIL.Image as Image

in_fn = '/media/data_cifs/nsf_levels/TURK_IMS/trainset/n12103894_3838.jpg'
out_fn = '/media/data_cifs/nsf_levels/TURK_IMS/trainset_bw/n12103894_3838.jpg'

img = Image.open(in_fn)
# convert to Grayscale
im_mat = np.asarray(img.convert('L'))
# save
im_tosave = Image.fromarray(im_mat).convert('L')
im_tosave.save(out_fn)
