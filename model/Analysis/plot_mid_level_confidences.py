#!/usr/bin/env python

from rapid_categorization.model import util
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Data # import data_loader # Changed this from Jonah script [changed line 1 of 2]

def mid_mod_ims(data, model_name, num, classifier_type, train_batches, set_index, set_name):
    n_layers = 15
    # Calculate stats per image for sample populataion
    model_comp = {} # data for model comparison
    im2lab = Data.im2lab # im2lab = data.im2lab # Changed this from Jonah script [changed line 2 of 2] ***Check 'data' in mid_mod_ims
    im2key = {} # maps image to index in model comp array
    im2key_ana = {}
    animal_ind = 0
    nonanimal_ind =0
    layer_names = util.get_model_layers(model_name)
    # store image names with index in dict to hold model comparisons
    for im in im2lab.keys():
        if im2lab[im] == 'animal':
            im2key[im] = animal_ind
            im2key_ana[im] = animal_ind
            animal_ind +=1
        else:
            im2key[im] = nonanimal_ind
            im2key_ana[im] = nonanimal_ind+150
            nonanimal_ind +=1
    #Easy Hard for Model
    model_comp['animal'] = np.ones([1,animal_ind])
    model_comp['nonanimal'] = np.ones([1,animal_ind])
    model_dist = {}
    model_dist['animal'] = np.ones([n_layers,animal_ind])
    model_dist['nonanimal'] = np.ones([n_layers,nonanimal_ind])
    modeldata = None
    for i_layer,layer_name in enumerate(layer_names):
        # calculate features
        modeldata = np.load(util.get_predictions_filename(model_name, layer_name, classifier_type=classifier_type, train_batches=train_batches, set_index=set_index, set_name=set_name))
        for index in range(len(modeldata['source_filenames'])):
            impath = modeldata['source_filenames'][index]
            imname = impath.split('/')[-1]+'.jpg'
            model_dist[im2lab[imname]][i_layer][im2key[imname]] = modeldata['hyper_dist'][index]
    inds = [layer_names.index('conv4_2ex'), layer_names.index('conv4_3ex'),layer_names.index('conv5_1ex'),layer_names.index('conv5_2ex')]
    for index in range(len(modeldata['source_filenames'])):
        temp = []
        impath = modeldata['source_filenames'][index]
        imname = impath.split('/')[-1]+'.jpg'
        for ind in inds:
            if im2lab[imname] == 'animal':
                temp.append(model_dist[im2lab[imname]][ind,im2key[imname]])
            else:
                d = model_dist[im2lab[imname]][ind,im2key[imname]]*(-1)
                temp.append(d)

        model_comp[im2lab[imname]][0][im2key[imname]] = np.mean(temp)

    mc_a_inds = np.argsort(model_comp['animal'][0])
    mc_na_inds = np.argsort(model_comp['nonanimal'][0])



    key2im = {}
    key2im['animal'] = {}
    key2im['nonanimal'] = {}
    for index in range(len(modeldata['source_filenames'])):
        temp = []
        impath = modeldata['source_filenames'][index]
        imname = impath.split('/')[-1]+'.jpg'
        cat = im2lab[imname]
        key = im2key[imname]
        key2im[cat][key] = imname

    top_a = []
    bottom_a = []
    top_na = []
    bottom_na = []


    for im in range(num):
        a = key2im['animal'][mc_a_inds[im]]
        bottom_a.append(a)
        na = key2im['nonanimal'][mc_na_inds[im]]
        bottom_na.append(na)

    for im in range(-1,-1-num,-1):
        a = key2im['animal'][mc_a_inds[im]]
        top_a.append(a)
        na = key2im['nonanimal'][mc_na_inds[im]]
        top_na.append(na)
#
#    #Plot easy animal
#    row = 1
#    col = 8
#    count = 0
#    f, subp =  plt.subplots(row, col,squeeze=False)
#    for r in range(row):
#        for c in range(col):
#            im_name = top_a[count]
#            count +=1
#            im_path = os.path.join(base_path,im_name)
#            imn = im_name.split('.')[0]+'.jpg'
#            img = Image.open(im_path)
#            im_mat = np.asarray(img.convert('L'))
#            subp[r][c].imshow(im_mat,cmap='gray')
#            subp[r][c].get_xaxis().set_ticks([])
#            subp[r][c].get_yaxis().set_ticks([])
#            subp[r][c].set_title(imn)
#
#            hypd = model_comp['animal'][0][im2key[imn]]
#            subp[r][c].set_xlabel('Model Conf: '+str(np.round(hypd,1)))
#    f.suptitle('Easiest Animals for VGG16 Intermediate Levels')
#
#     #Plot hard animal
#    row = 1
#    col = 8
#    count = 0
#    f, subp =  plt.subplots(row, col,squeeze=False)
#    for r in range(row):
#        for c in range(col):
#            im_name = bottom_a[count]
#            count +=1
#            im_path = os.path.join(base_path,im_name)
#            imn = im_name.split('.')[0]+'.jpg'
#            img = Image.open(im_path)
#            im_mat = np.asarray(img.convert('L'))
#            subp[r][c].imshow(im_mat,cmap='gray')
#            subp[r][c].get_xaxis().set_ticks([])
#            subp[r][c].get_yaxis().set_ticks([])
#            subp[r][c].set_title(imn)
#
#            hypd = model_comp['animal'][0][im2key[imn]]
#            subp[r][c].set_xlabel('Model Conf: '+str(np.round(hypd,1)))
#    f.suptitle('Hardest Animals for VGG16 Intermediate Levels')
#
#    #Plot easy nonanimal
#    row = 1
#    col = 8
#    count = 0
#    f, subp =  plt.subplots(row, col,squeeze=False)
#    for r in range(row):
#        for c in range(col):
#            im_name = top_na[count]
#            count +=1
#            im_path = os.path.join(base_path,im_name)
#            imn = im_name.split('.')[0]+'.jpg'
#            img = Image.open(im_path)
#            im_mat = np.asarray(img.convert('L'))
#            subp[r][c].imshow(im_mat,cmap='gray')
#            subp[r][c].get_xaxis().set_ticks([])
#            subp[r][c].get_yaxis().set_ticks([])
#            subp[r][c].set_title(imn)
#
#            hypd = model_comp['nonanimal'][0][im2key[imn]]
#            subp[r][c].set_xlabel('Model Conf: '+str(np.round(hypd,1)))
#    f.suptitle('Easiest Non-Animals for VGG16 Intermediate Levels')
#
#     #Plot hard nonanimal
#    row = 1
#    col = 8
#    count = 0
#    f, subp =  plt.subplots(row, col,squeeze=False)
#    for r in range(row):
#        for c in range(col):
#            im_name = bottom_na[count]
#            count +=1
#            im_path = os.path.join(base_path,im_name)
#            imn = im_name.split('.')[0]+'.jpg'
#            img = Image.open(im_path)
#            im_mat = np.asarray(img.convert('L'))
#            subp[r][c].imshow(im_mat,cmap='gray')
#            subp[r][c].get_xaxis().set_ticks([])
#            subp[r][c].get_yaxis().set_ticks([])
#            subp[r][c].set_title(imn)
#
#            hypd = model_comp['nonanimal'][0][im2key[imn]]
#            subp[r][c].set_xlabel('Model Conf: '+str(np.round(hypd,1)))
#    f.suptitle('Hardest Non-Animals for VGG16 Intermediate Levels')

   #plot background accuracy
    lvs = []
    confids = []

    for cat in model_dist:
        for lev in range(n_layers):
            for im in range(len(model_dist[cat][lev])):
                lvs.append(lev)
                if cat == 'animal':
                    confids.append(model_dist[cat][lev,im])
                else:
                    d = (-1)*model_dist[cat][lev,im]
                    confids.append(d)

    # Plot of accuracy accross levels with these images
    model_dist['nonanimal'] = model_dist['nonanimal']*(-1)
    combo_easy  = np.concatenate((top_a,top_na))
    combo_hard  = np.concatenate((bottom_a,bottom_na))
    mod_perf_easy = np.ones([n_layers,len(combo_easy)])
    for lyr in range(n_layers):
        for i in range(len(combo_easy)):
            im = combo_easy[i]
            key = im2key[im]
            mod_perf_easy[lyr,i] = model_dist[im2lab[im]][lyr][im2key[im]]
    mod_perf_hard = np.ones([n_layers,len(combo_hard)])
    for lyr in range(n_layers):
        for i in range(len(combo_hard)):
            im = combo_hard[i]
            key = im2key[im]
            mod_perf_hard[lyr,i] = model_dist[im2lab[im]][lyr][im2key[im]]

    fig, ax1 = plt.subplots()
    ax1.set_ylim([-2,5])
    ax1.plot([-0.5,n_layers-0.5],[0,0],color='0',ls='-')
    ax1.scatter(lvs,confids,color='0.8',label='All Images')
    ax1.plot([7.8,7.8],[-2,5],color='0.5',ls='--')
    ax1.plot([11.2,11.2],[-2,5],color='0.5',ls='--', label='Intermediate Levels')
    ax1.plot(xrange(n_layers),np.mean(mod_perf_easy,1),'go', label='Easiest for Mid-Layers')
    ax1.set_ylabel('Model Confidence')
    plt.xticks(rotation= 5)
    p1_fit = np.polyfit(xrange(n_layers),np.mean(mod_perf_easy,1),3)
    p1_fn = np.poly1d(p1_fit)
    xs = np.linspace(0, n_layers-1)
    ax1.plot(xs,p1_fn(xs),'g')

    ax1.plot(xrange(n_layers),np.mean(mod_perf_hard,1),'yo', label='Hardest for Mid-Layers')

    p2_fit = np.polyfit(xrange(n_layers),np.mean(mod_perf_hard,1),3)
    p2_fn = np.poly1d(p2_fit)
    xs = np.linspace(0, n_layers-1)
    ax1.plot(xs,p2_fn(xs),'y')

    ax1.set_xlim([-0.5, n_layers-0.5])
    plt.xticks(range(n_layers),util.get_model_layers(model_name, short=True),rotation=70)
    plt.sca(ax1)
    ax1.set_xlabel('Layer (Increasing Complexity)')
    plt.title('Model Confidence in Mid-Layer Extreme Images (%s)' % model_name)
    ax1.legend(loc="upper left")

if __name__ == '__main__':
    mid_mod_ims('VGG16', 12, 'svm', range(16), 0, 'turk')
    plt.show()
