#!/usr/bin/env python
import os
import sys
import operator as op
import predict
import util
from ext_caffe_direct import ext_caffe_direct

def opt_C(cvals, model_name, feature_name, classifier_type, set_name, train_batch_num, do_norm):
    import train_classifier
    val_batches = range(16,20)
    C_res = {}
    for C in cvals:
        props = dict()
        props['C'] = C
        clf_fn_c = util.get_classifier_filename(model_name, feature_name, classifier_type, range(train_batch_num),
                                                set_name=set_name, do_norm=do_norm, C=C)
        if os.path.isfile(clf_fn_c):
            norm_fn = util.get_norm_filename(model_name, feature_name, 0, set_name)
            C_res[str(C)] = predict.get_accuracy_for_sets(model_name, feature_name, set_name, val_batches, norm_fn, clf_fn_c, do_norm)
            print 'Acc %2.2f in %s' % (C_res[str(C)]*100, clf_fn_c)
        else:
            C_res[str(C)] = train_classifier.train_classifier(model_name, feature_name, classifier_type, props, set_name, train_batch_num, do_norm, val_batches)
    best_C = int(max(C_res.iteritems(), key=op.itemgetter(1))[0])
    print 'Optimal C Value: ' + str(best_C) + ', ' + str(C_res[str(best_C)]) + ' accuracy'
    return best_C


def build_classifiers(opt_params, batch_size, imset, extract_trainset, best_C, random_sample_ident, do_norm = True, force_overwrite = False, gpu_idx=0):
    """ Runs the variety of scripts necessary to extract layers
    """

    # Unpack Parameters
    model_name = opt_params['model_name']
    feature_name = opt_params['feature_name']
    set_name = opt_params['set_name']
    classifier_type = opt_params['classifier_type']
    train_batch_num = opt_params['train_batch_num']

    # Extract layer
    rand_samp = None
    if extract_trainset == True:
        test_batches = range(0,20)+imset
    else:
        test_batches = imset
    extract_layers(model_name=model_name, feature_name=feature_name, random_sample_ident=random_sample_ident, set_name=set_name, batches=test_batches, batch_size=batch_size, gpu_idx=gpu_idx, force_overwrite=force_overwrite)

    # predict
    train_batches = range(0, train_batch_num)

    clf_fn = util.get_classifier_filename(model_name, feature_name, classifier_type, train_batches, set_name=set_name, do_norm=do_norm)
    norm_fn = util.get_norm_filename(model_name, feature_name, train_batches[0], set_name)

    if not os.path.exists(clf_fn):
        print 'Creating classifier %s...' % clf_fn
        # optimize C
        if best_C is None:
            best_C = opt_C(opt_params['cvals'], model_name, feature_name, classifier_type, set_name, train_batch_num, do_norm)
        else:
            best_C = opt_C([best_C], model_name, feature_name, classifier_type, set_name, train_batch_num, do_norm)
        # Link optimized classifier name to regular one
        clf_fn_c = util.get_classifier_filename(model_name, feature_name, classifier_type, range(train_batch_num), set_name=set_name, do_norm=do_norm, C=best_C)
        os.link(clf_fn_c, clf_fn)
    else:
        print 'Using existing classifier at %s' % clf_fn

    extract_predictions(opt_params, set_name, test_batches, do_norm=do_norm, force_overwrite=force_overwrite)
    print "Extraction, Optimization, Training, and Prediction complete for "+feature_name


def extract_layers(model_name, feature_name, random_sample_ident, set_name, batches, batch_size, gpu_idx, force_overwrite=False):
    # Extract layer
    rand_samp = None
    for i in batches:
        input_root = util.get_input_image_root(i, set_name)
        input_file = util.get_imageset_filename_raw(i, set_name)
        output_file = util.get_feature_filename(model_name, feature_name, i, set_name)
        if os.path.exists(output_file) and not force_overwrite:
            print 'Extraction: Skipping set %d at %s' % (i, output_file)
            continue
        print 'Extracting from set %d to %s...' % (i, output_file)
        ext_samp = ext_caffe_direct(input_root, input_file, output_file, model_name, feature_name, random_sample_ident, rand_samp, batch_size, gpu_idx=gpu_idx, set_name=set_name)
        if rand_samp is None: rand_samp = ext_samp


def extract_predictions(opt_params, test_set, test_batches, do_norm, force_overwrite=False):
    model_name = opt_params['model_name']
    feature_name = opt_params['feature_name']
    set_name = opt_params['set_name']
    classifier_type = opt_params['classifier_type']
    train_batch_num = opt_params['train_batch_num']
    train_batches = range(0, train_batch_num)

    clf_fn = util.get_classifier_filename(model_name, feature_name, classifier_type, train_batches, set_name=set_name, do_norm=do_norm)
    norm_fn = util.get_norm_filename(model_name, feature_name, train_batches[0], set_name)

    for i_batch,test_batch in enumerate(test_batches):
        data_fn = util.get_feature_filename(model_name, feature_name, test_batch, test_set)
        source_fn = util.get_imageset_filename(test_batch, test_set)
        out_fn = util.get_predictions_filename(model_name, feature_name, classifier_type, train_batches, test_batch, test_set)
        if not force_overwrite:
            if os.path.isfile(out_fn):
                print 'Skipping %s.' % out_fn
                continue
        print 'Predicting %s.' % out_fn
        predict.do_predictions(data_fn, source_fn, norm_fn, clf_fn, out_fn, do_norm)


if __name__ == '__main__':
    # Set extraction parameters
    batch_size = 10
    extract_trainset = True
    random_sample_ident = True
    # Set SVM parameters
    should_opt = False
    opt_params = {}
    # opt_params['set_name'] = 'dist_vehicles_50'
    # opt_params['set_name'] = 'dist_vehicles_25'
    # opt_params['set_name'] = 'struct_vehicles_25'
    # opt_params['set_name'] = 'artifact_vehicles'
    opt_params['set_name'] = 'artifact_dist_vehicle_target' #<<<<---
    # opt_params['set_name'] = 'artifact_sport_vehicles'
    # opt_params['set_name'] = 'artifact_vehicles_human_test' # <<<<---
    opt_params['cvals'] = range(-5,-2)
    opt_params['model_name'] = 'VGG16'
    opt_params['feature_name'] = 'fc7ex'
    opt_params['classifier_type'] = 'svm'
    opt_params['train_batch_num'] = 16
    default_C = -4
    opt_params['subset'] = ''#'_pca1'

    #build_classifiers(opt_params, batch_size, [], extract_trainset, best_C=None, random_sample_ident=random_sample_ident, force_overwrite=False)
    imset = [120, 130, 140]#1603241729
    extract_layers(
        opt_params['model_name'],
        opt_params['feature_name'],
        random_sample_ident=random_sample_ident,
        set_name='clicktionary',
        batches=imset,
        batch_size=batch_size, gpu_idx=0,
        force_overwrite=False)
    extract_predictions(opt_params,
        'clicktionary',
        imset, do_norm=True,
        force_overwrite=False)


    # Lines for Michele to run
    # feature_names_for_VGG16 = ['conv1_1ex', 'conv1_2ex', 'conv2_1ex', 'conv2_2ex', 'conv3_1ex', 'conv3_2ex', 'conv3_3ex', 'conv4_1ex', 'conv4_2ex', 'conv4_3ex', 'conv5_1ex', 'conv5_2ex', 'conv5_3ex', 'fc6ex','fc7ex']
    # for feat_name in feature_names_for_VGG16:
    #     opt_params['feature_name'] = feat_name
    #     # build_classifiers(opt_params, batch_size, [], extract_trainset, best_C=None, random_sample_ident=random_sample_ident, force_overwrite=False)
    # # imset = [70]#1603241729
    #     range_batches = range(16)
    #     extract_layers(opt_params['model_name'], opt_params['feature_name'], random_sample_ident=random_sample_ident, set_name='artifact_vehicles_turk', batches=range_batches, batch_size=batch_size, gpu_idx=0, force_overwrite=False)
    #     extract_predictions(opt_params, 'artifact_vehicles_turk', range_batches, do_norm=True, force_overwrite=False)
