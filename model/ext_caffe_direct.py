#!/usr/bin/env python

import util
import pickle
import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from scipy import misc

# Run image list through caffe model and extract features
def ext_caffe_direct(
        input_root,
        input_file,
        output_file,
        model_name,
        feature_name,
        random_sample_ident,
        rand_samp,
        batch_size=1,
        flatten_features=True,
        gpu_idx=0,
        set_name=None):
    if gpu_idx == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_idx)
    rand_subset = rand_samp
    model_file = util.get_model_filename(model_name, feature_name)
    print model_file
    mean_file = util.get_mean_filename(model_name)
    crop_size = util.get_crop_size(model_name)
    weights_file = util.get_weights_filename(model_name)
    # Load image filenames
    print input_file
    with open(input_file) as fid:
        input_lines = fid.read().splitlines()
        inputs = [s.split('\t')[0] for s in input_lines]
    n = len(inputs)
    n_batches = (n-1) / batch_size + 1
    print 'Processing %d images in %d batches.' % (n, n_batches)
    # Init caffe
    if weights_file is None:
        net = caffe.Net(model_file, caffe.TEST)
    else:
        net = caffe.Net(model_file, weights_file, caffe.TEST)
    data_shape = net.blobs['data'].data.shape
    net.blobs['data'].reshape(batch_size,data_shape[1],data_shape[2],data_shape[3])
    # Init input transformer
    transformer = caffe.io.Transformer({'data': data_shape})
    transformer.set_transpose('data', (2,0,1))
    mean = np.load(mean_file) if mean_file is not None else None
    if crop_size is not None:
        if mean is not None:
            if mean.shape[1] > crop_size[0]:
                y0 = int((mean.shape[1] - crop_size[0])/2)
                mean = mean[:,y0:y0+crop_size[0],:]
            if mean.shape[2] > crop_size[1]:
                x0 = int((mean.shape[2] - crop_size[1])/2)
                mean = mean[:,:,x0:x0+crop_size[1]]
        if set_name != 'clicktionary':
            transformer.set_crop('data', [crop_size[0], crop_size[1]])
    if mean is not None: transformer.set_mean('data', mean)
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has chann
    # Process batches
    ii = 0
    for i in xrange(n_batches):
        nb = min(batch_size, n - ii)
        # Load images
        for j in xrange(nb):
            fn = inputs[ii+j]
            if not fn.endswith('.png'):
                fn = os.path.join(input_root, fn + '.jpg')
            img = caffe.io.load_image(fn)
            if set_name == 'clicktionary':
                img = misc.imresize(img, crop_size)
            net.blobs['data'].data[j,...] = transformer.preprocess('data', img)
        net.forward()
        output = net.blobs[feature_name].data
        output = np.reshape(output[...], (nb, -1))  # flatten
        if random_sample_ident is not None:
            rand_sample_size = 4096
            n_features = output.shape[1]
            if rand_subset is None:
                random_sample_filename = util.get_random_sample_filename(model_name, feature_name, random_sample_ident)
                if os.path.isfile(random_sample_filename):
                    print 'Loading random sample from %s' % random_sample_filename
                    rand_subset = pickle.load(open(random_sample_filename, 'rb'))
                elif n_features > rand_sample_size:
                    print 'Creating new random sample into %s' % random_sample_filename
                    rand_subset = np.random.choice(range(np.size(output,1)),rand_sample_size)
                    pickle.dump(rand_subset,open(random_sample_filename,'wb'))
                else:
                    print 'Random subset not needed: Insuficient output feature count.'
                    random_sample_ident = None
            if rand_subset is not None:
                output = output[:,rand_subset] #random sample


        if not i:
            all_output_shape = list(output.shape[:])
            all_output_shape[0] = n
            all_output = np.zeros(all_output_shape)
        all_output[ii:ii+nb,...] = output[...]
        ii += nb
        if not ((i+1) % 10):
            if not ((i+1) % 100):
                print '%d' % (i+1)
            else:
                print '%d' % (i+1),
        else:
            print '.',
    # Save outputs
    if flatten_features:
        all_output = np.reshape(all_output, (n, -1))
    np.savez(output_file, data=all_output)
    return rand_subset
