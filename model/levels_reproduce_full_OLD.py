# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import operator as op
import predict
import util
import pickle
import datetime
import numpy as np
from PIL import Image
import leargist


def ext_caffe_direct(input_root, input_file, output_file, feature, model_file,should_random_sample, sample_path,rand_samp, set_num, weights_file=None, batch_size=1, mean_file=None, crop_size=None, flatten_features=True):
    import caffe
    import numpy as np
    from copy import deepcopy
    import os
    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    rand_subset = rand_samp
    # Load image filenames
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
            img = caffe.io.load_image(os.path.join(input_root, inputs[ii+j]))
            net.blobs['data'].data[j,...] = transformer.preprocess('data', img)
        net.forward()
        output = net.blobs[feature].data
        if should_random_sample == True:
            output = np.reshape(output[...], (nb,-1)) #flatten
            if rand_subset==None:
                rand_subset = np.random.choice(range(np.size(output,1)),4096)
                now=datetime.datetime.now()
                ident = str(now.month)+str(now.day)+str(now.hour)+str(now.minute)
                pickle.dump(rand_subset,open(os.path.join(sample_path,'randomsubsets'+ident+'.p'),'wb'))
            output = output[:,rand_subset] #random sample

            
        if not i:
            all_output_shape = list(output.shape[:])
            all_output_shape[0] = n
            all_output = np.zeros(all_output_shape)
        all_output[ii:ii+nb,...] = output[...]
        ii += nb
        if not (i % 10):
            print '%d' % i,
        else:
            print '.',
    # Save outputs
    if flatten_features:
        all_output = np.reshape(all_output, (n, -1))
    np.savez(output_file, data=all_output)
    return rand_subset
    

def do_gist(batch_name, base_npz_dir,outfile,input_file,input_root):
    # Basic settings
    base_path = base_npz_dir
    print 'Running GIST extraction for %s. Base path %s' % (batch_name, base_path)
    out_pth = os.path.join(base_path, 'gist')
    out_fn = outfile %(batch_name)
    if os.path.isfile(out_fn):
        print "...skip."
        return
    filelist_fn = input_file %(batch_name)

    # load file list to process
    filelist_data = open(filelist_fn, 'rt').readlines()
    filenames = [s.split('\t')[0] for s in filelist_data]
    true_labels = [s.split('\t')[1].replace('\n','') for s in filelist_data]

    n_files = len(filenames)
    outputs = None
    for i_file, filename in enumerate(filenames):
        full_fn = os.path.join(input_root, filename)
        img = Image.open(full_fn)
        data = leargist.color_gist(img)
        if outputs is None:
            outputs = np.zeros((n_files, data.size), 'float32')
        outputs[i_file,:] = data
        print "Done file %05d of %05d: %s" % (i_file, n_files, full_fn)

    # output feature vector
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    print 'Saving to: %s' % out_fn
    np.savez(out_fn, data=outputs)


    
def opt_C(cvals,feature_name,classifier_type, train_batch_num,base_npz_dir, props):
    import train_classifier
    do_norm = True
    test_batches = range(20,30)
    return train_classifier.train_classifier(feature_name, classifier_type, props, train_batch_num, do_norm, base_npz_dir, test_batches)


def levels_rep(input_root,infile,outfile, feature, model_file,weights_file,batch_size, mean_file, crop_size, imset, extract_trainset, opt_params, should_opt, default_C,should_random_sample):
    """ Runs the variety of scripts necessary to extract layers
    """
    sample_path = os.path.join(opt_params['base_npz_dir'],opt_params['feature_name'])
    # Set PYTHON and CUDA paths 
    sys.path.append('/home/sven2/s2caffe/python')
    sys.path.append('/home/sven2/python/hmax_orig')
    os.environ['LD_LIBRARY_PATH'] = '/home/sven2/python/hmax_orig'
    if not(feature == 'gist'):
        # Extract layer
        rand_samp = None
        if extract_trainset == True:
            setrange = range(0,30)+[imset]
        else:
            setrange = [imset]
        for i in setrange:
            input_file = infile % i
            output_file = outfile % i
            print 'Extracting from set '+str(i)
            if rand_samp ==None:
                rand_samp = ext_caffe_direct(input_root, input_file, output_file, feature, model_file, should_random_sample,sample_path,rand_samp, i,weights_file, batch_size, mean_file, crop_size)    
            else:
                ext_caffe_direct(input_root, input_file, output_file, feature, model_file, should_random_sample,sample_path,rand_samp, i,weights_file, batch_size, mean_file, crop_size) 
    else:
        if extract_trainset == True:
            setrange = range(5,10)+[imset]
        else:
            setrange = [imset]
        for i in setrange:        
            batch_name = i
            do_gist(batch_name,opt_params['base_npz_dir'],outfile, infile,input_root)

    feature_name = opt_params['feature_name'] 
    classifier_type = opt_params['classifier_type'] 
    train_batch_num = opt_params['train_batch_num'] 
    base_npz_dir = opt_params['base_npz_dir']
    best_C = -3
       
    # optimize C 
#    if should_opt == True:
#        C_res = {}
#        cvals = opt_params['cvals']
#        feature_name = opt_params['feature_name'] 
#        classifier_type = opt_params['classifier_type'] 
#        train_batch_num = opt_params['train_batch_num'] 
#        base_npz_dir = opt_params['base_npz_dir']
#        for C in cvals:
#            props = dict()
#            props['C'] = C
#            C_res[str(C)] = opt_C(cvals,feature_name,classifier_type, train_batch_num,base_npz_dir, props)
#        
#        
#        best_C = max(C_res.iteritems(), key=op.itemgetter(1))[0]
#        print 'Optimal C Value: '+str(best_C)+', '+str(C_res[str(best_C)])+' accuracy'
#    else: 
#        best_C = default_C
#        print "Using Default C Value: "+str(best_C)
    
    # predict
    feature_name = opt_params['feature_name']
    classifier_type = classifier_type+'NC'+str(best_C)
    subset = opt_params['subset']
    force_overwrite = True
    base_path = util.get_base_path()
    
    train_batches = range(0, 20)
    test_batches = range(500,501)
    set_name = 'setb50k'
    do_norm = True
    clf_fn = os.path.join(base_npz_dir, feature_name, '%s_%s%s_%d-%d.pickle' % (classifier_type, set_name, subset, train_batches[0], train_batches[-1]))
    norm_fn = os.path.join(base_npz_dir, feature_name, 'norm_%s%s_%d.pickle' % (set_name, subset, train_batches[0]))
    out_pth = os.path.join(base_npz_dir, 'predictions', '%s_%s_%s%s_%d-%d' % (feature_name, classifier_type, set_name, subset, train_batches[0], train_batches[-1]))
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)

    for i_batch,test_batch in enumerate(test_batches):
        data_fn = os.path.join(base_npz_dir, feature_name, '%s_%05d%s.npz' % (set_name, test_batch, subset))
        source_fn = os.path.join(base_path, 'sets/%s_%05d.txt' % (set_name, test_batch))

        out_fn = os.path.join(out_pth, '%05d.npz' % (test_batch))
        if not force_overwrite:
            if os.path.isfile(out_fn):
                print 'Skipping %s.' % out_fn
                continue
        print 'Predicting %s.' % out_fn
        predict.do_predictions(data_fn, source_fn, norm_fn, clf_fn, out_fn, do_norm)
    
    print "Extraction, Optimization, Training, and Prediction complete for "+opt_params['feature_name']



# Set extraction parameters
imset = 500
input_root= '/media/data_cifs/sven2/ccv/AnNonAn_cc256/'
input_file = '/media/data_cifs/sven2/ccv/AnNonAn_cc256/sets/setb50k_%05d.txt'
output_file = '/media/data/nsf_levels/conv5_1/setb50k_%05d.npz'
weights_file = '/media/data/nsf_levels/VGG_ILSVRC_16_layers.caffemodel'
# weights_file = '/media/data/nsf_levels/VGG_ILSVRC_19_layers.caffemodel' #VGG19
model_file = 'deploy_VGG16_conv5_1.prototxt'
batch_size = 15
feature = 'conv5_1ex'
mean_file = '/home/sven2/s2caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
crop_size = (224, 224)
extract_trainset = True
should_random_sample = True
# Set SVM parameters
should_opt = True
opt_params = {}
opt_params['cvals'] = range(-5,6)
opt_params['feature_name'] = 'VGG16_conv5_1ex'
opt_params['classifier_type'] = 'svm'
opt_params['train_batch_num'] = 20
opt_params['base_npz_dir'] = '/media/data/nsf_levels/'
default_C = 0
opt_params['subset'] = ''#'_pca1'
print feature
levels_rep(input_root, input_file, output_file, feature, model_file, weights_file, batch_size,mean_file, crop_size, imset, extract_trainset,opt_params, should_opt,default_C, should_random_sample)
