#!/usr/bin/env python

print "Hello, this is ext_hmax_features.py.\n"

# Extract hmax features for one batch of animal/nonanimal images

import os
import sys
import pickle
import numpy as np
from rapid_categorization.models import legacy
from pycuda import gpuarray
import scipy as sp
import scipy.io as sio
from rapid_categorization import tools
from scipy import fliplr, flipud

# Settings from command line
if len(sys.argv) < 5:
    custom_dict = None
else:
    custom_dict = sys.argv[4]
    max_dict_words = 1024

if len(sys.argv) < 4:
    i_gpu = 0
else:
    i_gpu = int(sys.argv[3])

if len(sys.argv) < 3:
    batch_name = 'setb50k_00000'
else:
    batch_name = 'setb50k_%05d' % int(sys.argv[2])

if len(sys.argv) < 2:
    ext_feature = 'c2rbf'
else:
    ext_feature = sys.argv[1]

s2_is_rbf = False
out_fn_suffix = ''
if ext_feature == 'c1':
    do_c2 = False
elif ext_feature == 'c2':
    do_c2 = True
elif ext_feature == 'c2rbf':
    do_c2 = True
    s2_is_rbf = True
    s2_rbf_sigma = 1.0/3
    out_fn_suffix = 'rbf'
else:
    raise RuntimeError('Unknown feature: %s' % ext_feature)

# Basic settings
base_path = '/users/seberhar/data/data/AnNonAn_cc256'
if not os.path.exists(base_path):
    base_path = '/home/sven2/DeleteMe/convtest'
    i_gpu = 1
if custom_dict is None:
    feature_prefix = 'hmax_'
else:
    feature_prefix = 'hmax%s_' % custom_dict
categories = ['NON-ANIMAL', 'ANIMAL']
max_dict_words = 1e10

print 'Running batch %s for feature %s on gpu %d. Base path %s' % (batch_name, ext_feature, i_gpu, base_path)

save_padding_info = False
if do_c2:
    padding_fn = os.path.join(base_path, 'padding_info_c2.pickle')
else:
    padding_fn = os.path.join(base_path, 'padding_info_c1.pickle')
if custom_dict is None:
    dict_fn = os.path.join(base_path, 'hmax_c1', 'dict4d.pickle')
elif custom_dict == 'multiscale':
    dict_fn = '../models/legacy/data/dict_multiscale.mat'
else:
    raise RuntimeError('dictionary unknown')

if do_c2:
    rval_names = ['c2']
else:
    rval_names = ['c1']
n_rvals = len(rval_names)

out_pth = []
out_fn = []
has_missing_files = False
for i in xrange(n_rvals):
    out_pth += [os.path.join(base_path, feature_prefix + rval_names[i] + out_fn_suffix)]
    this_out_fn = os.path.join(out_pth[i], batch_name + '.npz')
    out_fn += [this_out_fn]
    if not os.path.isfile(this_out_fn):
        has_missing_files = True

if not has_missing_files:
    print 'Output files already exists. Aborting.'
    exit()

filelist_fn = os.path.join(base_path, 'sets', batch_name + '.txt')

# load file list to process
filelist_data = open(filelist_fn, 'rt').readlines()
filenames = [s.split('\t')[0] for s in filelist_data]
true_labels = [s.split('\t')[1].replace('\n','') for s in filelist_data]

# Find indices of contents by looking for extents of padding value in results pyramid
def get_padding_indices(layer, padding_value):
    sz = layer.shape
    if (len(sz) < 4):
        return [], layer.size # no features
    padding_indices = []
    n_scales = sz[0]
    n_features = sz[1]
    n_total_output = 0
    sz_y = sz[2]
    sz_x = sz[3]
    for i_scale in xrange(n_scales):
        data = layer[i_scale,0]
        good_idcs = np.where(data != padding_value)
        if len(good_idcs[0]) == 0:
            pad_idcs = [0,0,0,0]
        else:
            pad_idcs = [np.min(good_idcs[0]), np.max(good_idcs[0])+1, np.min(good_idcs[1]), np.max(good_idcs[1])+1]
        padding_indices += [pad_idcs]
        n_total_output += (pad_idcs[1] - pad_idcs[0]) * (pad_idcs[3] - pad_idcs[2]) * n_features
    #print 'pad=%s' % str(padding_indices)
    return padding_indices, n_total_output

def zero_padding(layer, padding_value):
    """docstring for remove_padding"""
    layer[layer == padding_value] = 0
    return layer

def flatten_layer(layer, padding_indices, n_total_output):
    """Create vector of length n_total_output from layer areas indexed by padding_indices"""
    result = np.zeros(n_total_output)
    if not len(padding_indices):
        result[:] = layer[:]
    else:
        idx = 0
        for i_layer in xrange(layer.shape[0]):
            pi =  padding_indices[i_layer]
            data = layer[i_layer,:,pi[0]:pi[1],pi[2]:pi[3]]
            n = data.size
            result[idx:idx+n] = data.flatten()
            idx += n
    return result

if do_c2:
    if dict_fn.split('.')[-1] == 'pickle':
        with open(dict_fn, 'rb') as fid:
            d = pickle.load(fid)
    else:
        d = sio.loadmat(dict_fn)
        d['n_words'] = len(d['w_sizes'])
    if d['n_words'] > max_dict_words:
        d['n_words'] = max_dict_words
        d['w_sizes'] = d['w_sizes'][:max_dict_words]
        d['l_words'] = d['l_words'][:max_dict_words]
    word_sizes = sorted(np.unique([s[0] for s in d['w_sizes']]))
    d["l_word_sizes"] = [[s,s] for s in word_sizes]
    max_word_size = np.max(word_sizes)
    d['l_words'] = d['l_words'][:,:,:max_word_size,:max_word_size]
    for i,ws in enumerate(word_sizes):
        d['l_words'][i,:,:ws,:ws] = flipud(fliplr(d['l_words'][i,:,:ws,:ws]))
    print 'Using dictionary with %d words with sizes %s.' % (d['n_words'], d["l_word_sizes"])
else:
    d = None

p       = legacy.params.ventral_2(d)
padding_value = p['model']['preprocessing']['pad_value']
img_new_size = p['model']['preprocessing']['img_base_size']
if s2_is_rbf:
    p['model']['s2']['function'] = 'extract_s2_rbf'
    p['model']['s2']['sigma'] = s2_rbf_sigma

n_scales = p['model']['preprocessing']['n_scales']
scaling_factor = p['model']['preprocessing']['scaling_factor']

# init the cuda kernels
ctx, s1_ker, c1_ker, s2_ker, c2_ker = legacy.ventral_pycuda.prepare_cuda_kernels(p, i_gpu)

# allocating memory on the gpu
s1_ = gpuarray.to_gpu(padding_value*sp.ones(p['model']['s1']['shape'], dtype = 'float32'))
c1_ = gpuarray.to_gpu(padding_value*sp.ones(p['model']['c1']['shape'], dtype = 'float32'))
s2_ = gpuarray.to_gpu(padding_value*sp.ones(p['model']['s2']['shape'], dtype = 'float32'))
c2_ = gpuarray.to_gpu(padding_value*sp.ones(p['model']['c2']['shape'], dtype = 'float32'))
layers_ = [s1_, c1_, s2_, c2_]

# moving the dictionary to the gpu
if do_c2:
    n_words = d['n_words']
    l_words = d['l_words'][:n_words] ; w_sizes = d['w_sizes'][:n_words]
    l_words_   = gpuarray.to_gpu(sp.array(l_words, order = 'C', dtype = 'float32'))
    w_sizes_   = gpuarray.to_gpu(sp.array(w_sizes, order = 'C', dtype = 'int32'))
else:
    l_words_ = []
    w_sizes_ = []
    p['model']['dictionary']['n_words'] = 0

padding_indices = None
n_total_outputs = None
outputs = [None] * n_rvals
n_files = len(filenames)

features = []
for i_file, filename in enumerate(filenames):
    full_fn = os.path.join(base_path, filename)
    # generate a pyramid from each frame
    imp, coor = tools.image.make_pyramid(full_fn, img_new_size, n_scales, scaling_factor, padding_value)

    # extract the hmax layers from the video
    out = dict()
    if do_c2:
        out['s1'], out['c1'], out['s2'], out['c2'] = legacy.ventral_pycuda.get_c2(imp, coor, l_words_, w_sizes_, p, s1_ker, c1_ker, s2_ker, c2_ker, layers_, debug = False)
    else:
        s1_, c1_ = legacy.ventral_pycuda.get_c1(imp, coor, p['model']['filters']['gabors'], p, s1_ker, c1_ker, layers_[0], layers_[1], debug = False)
        #out['s1'] = s1_.get()
        out['c1'] = c1_.get()
    rvals = [out[rval_name] for rval_name in rval_names]

    # Padding info init
    if padding_indices is None:
        padding_indices = []
        n_total_outputs = []
        for i in xrange(n_rvals):
            pi, nt = get_padding_indices(rvals[i], padding_value)
            padding_indices += [pi]
            n_total_outputs += [nt]
            outputs[i] = np.zeros((n_files, nt))
        if save_padding_info:
            with open(padding_fn, 'wb') as fid:
                pickle.dump(padding_indices, fid)
            print 'padding info saved to %s. aborting.' % padding_fn
            exit()

    # Flatten and store data
    for i in xrange(n_rvals):
        data = flatten_layer(rvals[i], padding_indices[i], n_total_outputs[i])
        print data
        outputs[i][i_file,:] = data

    print "Done file %05d of %05d: %s" % (i_file, n_files, full_fn)

# killing the cuda context
ctx.pop()

# output feature vector
for i in xrange(n_rvals):
    if not os.path.exists(out_pth[i]):
        os.makedirs(out_pth[i])
    #with open(out_fn[i], 'wb') as fid:
    #    pickle.dump(outputs[i], fid)
    print 'Saving to: %s' % out_fn[i]
    np.savez(out_fn[i], data=outputs[i])
