# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 13:50:44 2016

@author: jcader
"""

import levels_reproduce_full as lrf
import util
import sys
import os

def run_levels_expt(model_name, layers=None, gpu_idx=0):
    # Set extraction parameters
    turk_batches = [33, 34]#1603241729
    batch_size = 10
    extract_trainset = True
    random_sample_ident = True
    do_norm = True
    force_overwrite = False
    force_overwrite_turk = False
    # Set SVM parameters
    should_opt = False
    opt_params = {}
    opt_params['set_name'] = 'set'
    opt_params['cvals'] = range(-5, -2)
    opt_params['model_name'] = model_name
    opt_params['classifier_type'] = 'svm'
    opt_params['train_batch_num'] = 16
    best_C = None #-4
    opt_params['subset'] = ''  # '_pca1'
    if layers is None:
        layers = util.get_model_layers(model_name)
    for layer in layers:
        opt_params['feature_name'] = layer
        lrf.build_classifiers(opt_params=opt_params, batch_size=batch_size, imset=[], extract_trainset=extract_trainset, best_C=best_C, random_sample_ident=random_sample_ident, do_norm=do_norm, force_overwrite=force_overwrite, gpu_idx=gpu_idx)
        lrf.extract_layers(model_name, layer, random_sample_ident=random_sample_ident, set_name='turk', batches=turk_batches, batch_size=batch_size, gpu_idx=gpu_idx, force_overwrite=force_overwrite_turk)
        lrf.extract_predictions(opt_params, 'turk', turk_batches, do_norm, force_overwrite=force_overwrite_turk)

def run_levels(model_name, gpu_idcs, do_parallel):
    if do_parallel:
        layers = util.get_model_layers(model_name)
        for i,layer in enumerate(layers):
            os.system(' '.join(['/home/sven2/.virtualenvs/cv/bin/python', sys.argv[0], model_name, layer, str(gpu_idcs[i % len(gpu_idcs)]), '>'+util.at_log_path(model_name+'_'+layer), '2>&1', '&']))
    else:
        run_levels_expt(model_name, layers=None, gpu_idx=gpu_idcs[0])


if __name__ == '__main__':
    do_parallel = True
    if len(sys.argv) == 1:
        run_levels('VGG16_ft70000', [0, 3], do_parallel)
        #run_levels('AlexNet', [-1], do_parallel)
        run_levels('VGG19', [0, 3], do_parallel)
        run_levels('VGG16', [0, 3], do_parallel)
        #run_levels('VGG16_ft20000', [0, 3])
        #run_levels('VGG16_ft70000', [0, 3])
    else:
        run_levels_expt(sys.argv[1], layers=[sys.argv[2]], gpu_idx=int(sys.argv[3]))
