#!/usr/bin/env python

from rapid_categorization.levels import util
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import os
from rapid_categorization.levels import util
import shutil
from PIL import Image

def easy_hard_mod(data, model_name, layer_name):
    init_mode = False

    examples = [[],[]]
    for imt in ['animal', 'nonanimal']:
        human_res = data.model_comp[imt]['human_acc'][0]
        model_res = data.model_comp[imt][model_name + '_' + layer_name][0]
        if imt == 'nonanimal': model_res = model_res * -1
        human_ranks = np.argsort(-human_res)
        model_ranks = np.argsort(-model_res)
        # Find intersection of 100 extreme images for humans and model
        examples_ind = []
        examples_ind += [set(human_ranks[:150]) & set(model_ranks[-150:])]
        examples_ind += [set(human_ranks[-150:]) & set(model_ranks[:150])]
        print 'Class %s: Found %d easy for humans and %d easy for model.' % (imt, len(examples_ind[0]), len(examples_ind[1]))
        for j in (0,1):
            examples[j] += [(data.model_comp[imt]['source_images'][i][0],human_res[i],model_res[i]) for i in examples_ind[j]]

    tss = ('human participants', 'model')
    fns = ('Human', 'Model')
    for pth,imset,ii in zip(('human_examples', 'model_examples'), examples, (0,1)):
        f, subp = plt.subplots(1, 6, squeeze=False)
        c = 0
        pth = util.at_plot_path(pth)
        if not os.path.isdir(pth): os.makedirs(pth)
        for i,imdata in enumerate(imset):
            fn,hum,mod = imdata
            fn_full = '%s/%d.jpg' % (pth,i)
            if init_mode:
                srcpth = data.im2path[fn]
                shutil.copyfile(os.path.join(srcpth, fn), fn_full)
                continue
            if not os.path.isfile(fn_full): continue
            img = Image.open(fn_full)
            im_mat = np.asarray(img.convert('L'))
            subp[0][c].imshow(im_mat,cmap='gray')
            subp[0][c].get_xaxis().set_ticks([])
            subp[0][c].get_yaxis().set_ticks([])
            #subp[0][c].set_title(imn)
            subp[0][c].set_xlabel('H: %d%%\nM: %1.2f' % (hum, mod), fontsize=14)
            c += 1
            if c == 6: break
        f.suptitle('Easy for %s, Hard for %s' % (tss[ii], tss[1-ii]), y = 0.95, fontsize=15, weight='bold')
        f.set_size_inches(12, 2.6, forward=True)
        f.subplots_adjust(bottom=0.06, top=1, left=0.01, right=0.99)
        plt.savefig(util.at_plot_path(fns[ii] + '.png'))
        plt.savefig(util.at_plot_path(fns[ii] + '.pdf'))
