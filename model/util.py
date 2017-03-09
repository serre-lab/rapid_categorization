#!/usr/bin/env python

# Some Levels settings

import os
import numpy as np
import datetime
import platform

########################################################################################################################
# Path config

hostname = platform.node()
if hostname == 'x8':
    # Sven2's machine
    imageset_base_path = '/media/data_cifs/nsf_levels/'
    clicktionary_imageset_path = '/media/data_cifs/clicktionary/causal_experiment'
    base_path = '/media/data/nsf_levels/'
    log_path = '/media/data/nsf_levels/log'
    experiment_path = '/media/data_cifs/nsf_levels/Results'
    plot_path = '/media/data_cifs/nsf_levels/plots'
elif hostname.beginswith('gpu'):
    # CCV
    imageset_base_path = '/users/seberhar/data/data/AnNonAnNIPS'
    base_path = '/users/seberhar/data/data/AnNonAnNIPS'
    log_path = '/users/seberhar/data/data/AnNonAnNIPS/log'
else:
    raise RuntimeError('Unknown node. Please configure data paths.')


########################################################################################################################
# Output files
def at_log_path(log_filename):
    return os.path.join(log_path, log_filename)

def at_plot_path(plot_filename):
    return os.path.join(plot_path, plot_filename)

def get_model_plot_marker(model_name):
    if model_name == 'AlexNet': return 's'
    elif model_name == 'VGG16': return 'v'
    elif model_name.startswith('VGG16'): return '^'
    elif model_name.startswith('VGG19'): return 'o'
    else: return '.'

def get_model_plot_color(model_name, plot_type=None, brighten=False):
    if plot_type is None:
        if model_name == 'AlexNet': c = (0.0, 1.0, 0.0)
        elif model_name == 'VGG16': c = (1.0, 0.0, 0.0)
        elif model_name.startswith('VGG16'): c = (0.4, 0.0, 0.0)
        elif model_name.startswith('VGG19'): c = (0.0, 0.0, 1.0)
        else: c = (0.0, 0.0, 0.0)
    elif plot_type == 'acc':
        if model_name == 'AlexNet': c = (0.5, 0.5, 1.0)
        elif model_name == 'VGG16': c = (0.0, 0.0, 0.3)
        elif model_name.startswith('VGG16'): c = (0.0, 0.0, 1.0)
        elif model_name.startswith('VGG19'): c = (0.0, 0.5, 1.0)
        else: c = (0.0, 0.0, 0.0)
    elif plot_type == 'corr':
        if model_name == 'AlexNet': c = (1.0, 0.5, 0.5)
        elif model_name == 'VGG16': c = (0.3, 0.0, 0.0)
        elif model_name.startswith('VGG16'): c = (1.0, 0.0, 0.0)
        elif model_name.startswith('VGG19'): c = (1.0, 0.5, 0.0)
        else: c = (0.0, 0.0, 0.0)
    if brighten: c = [min(cc+0.8, 1.0) for cc in c]
    return c

def get_model_human_name(model_name, add_ft_label=True):
    # Cut off _ftXXXXXX, replace by (ft)
    parts = model_name.split('_')
    name = parts[0]
    if add_ft_label and (len(parts) > 1): name += ' (ft)'
    return name

########################################################################################################################
# Image source files
def at_imageset_base_path(subpath):
    return os.path.join(imageset_base_path, subpath)

def get_imageset_filename(set_index, set_name):
    if set_name=='turk': # Experiment set
        input_file = at_imageset_base_path('TURK_IMS/set1603241729_%d.txt' % set_index)
    elif set_name == 'clicktionary':
        input_file = os.path.join(clicktionary_imageset_path, 'images_%d.txt' % set_index)
    else:
        input_file = at_imageset_base_path('raw_ims/sets/train%s_%d.txt' % (set_name, set_index))
    return input_file

def get_imageset_filename_raw(set_index, set_name):
    if set_name == 'turk':
        input_file = at_imageset_base_path('TURK_IMS/raw_%s%d_0.txt' % (set_name, set_index))
    elif set_name == 'clicktionary':
        input_file = os.path.join(clicktionary_imageset_path, 'raw_images_%d.txt' % set_index)
    else:
        input_file = at_imageset_base_path('raw_ims/sets/raw_train%s_%d.txt' % (set_name, set_index))
    return input_file

def get_input_image_root(set_index, set_name):
    if set_name == 'turk':
        return at_imageset_base_path('TURK_IMS/set1603241729_%d' % set_index)
    else:
        return at_imageset_base_path('TURK_IMS/trainset_bw/')

def load_labels(set_name, batch_index):
    # Load labels for one batch of one image set
    source_fn = get_imageset_filename(batch_index, set_name)
    with open(source_fn, 'rt') as fid:
        source_data = fid.read().splitlines()
    true_labels = np.array([int(s.split('\t')[1]) for s in source_data])
    source_filenames = np.array([s.split('\t')[0] for s in source_data])
    return true_labels, source_filenames

def get_categories():
    return ['NON-ANIMAL', 'ANIMAL']


########################################################################################################################
# Data files

def get_base_path():
    return base_path

def at_base_path(subpath):
    return os.path.join(base_path, subpath)

def at_feature_path(model_name, feature_name, filename):
    feature_path = at_base_path('%s_%s' % (model_name, feature_name))
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    return os.path.join(feature_path, filename)

def get_feature_filename(model_name, feature_name, set_index, set_name='set'):
    return at_feature_path(model_name, feature_name, '%s_%d.npz' % (set_name, set_index))

def create_random_sample_filename(model_name, feature_name):
    # Ensure unique filename
    now = datetime.datetime.now()
    ident = str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    get_random_sample_filename(model_name, feature_name, ident)

def get_random_sample_filename(model_name, feature_name, ident):
    return at_feature_path(model_name, feature_name, 'randomsubsets' + str(ident) + '.p')

def get_classifier_filename(model_name, feature_name, classifier_type, train_batches, set_name='set', do_norm=True, C=None):
    classifier_paramstring = ''
    if do_norm: classifier_paramstring += 'N'
    if C is not None: classifier_paramstring += 'C%d' % C
    return at_feature_path(model_name, feature_name, '%s%s_%s_%d-%d.pickle' % (classifier_type, classifier_paramstring, set_name, train_batches[0], train_batches[-1]))

def get_norm_filename(model_name, feature_name, set_index, set_name='set'):
    return at_feature_path(model_name, feature_name, 'norm_%s_%s.pickle' % (set_name, set_index))

def get_predictions_filename(model_name, feature_name, classifier_type, train_batches, set_index, set_name='set'):
    prediction_path = at_base_path('predictions/%s_%s' % (model_name, feature_name))
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    prediction_subpath = os.path.join(prediction_path, '%s_%d-%d' % (classifier_type, train_batches[0], train_batches[-1]))
    if not os.path.exists(prediction_subpath):
        os.makedirs(prediction_subpath)
    return os.path.join(prediction_subpath, '%s_%d.npz' % (set_name, set_index))


########################################################################################################################
# Models

def at_model_path(subpath):
    return subpath

def get_model_filename(model_name, feature_name):
    model_name = model_name.split('_')[0] # Cut weights info from model name
    if feature_name.endswith('ex'): feature_name = feature_name[:-2] # cut 'ex' from feature name that is only used to label the extraction blob
    if model_name == 'AlexNet':
        return at_model_path('deploy_%s.prototxt' % (feature_name))
    else:
        return at_model_path('deploy_%s_%s.prototxt' % (model_name, feature_name))

def get_mean_filename(model_name):
    return '/home/sven2/s2caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

def get_weights_filename(model_name):
    if model_name == 'AlexNet': return '/home/sven2/s2caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    if model_name == 'VGG16': return at_base_path('VGG_ILSVRC_16_layers.caffemodel')
    if model_name.startswith('VGG16_ft'): return at_base_path(model_name + '.caffemodel')
    if model_name == 'VGG19': return at_base_path('VGG_ILSVRC_19_layers.caffemodel')
    return None

def get_crop_size(model_name):
    # Return input image crop size by model
    if model_name == 'AlexNet':
        return (227, 227)
    else:
        return (224, 224)

def get_model_layers(model_name, short=False):
    # Get names of available layers per model
    layers = []
    if model_name == 'AlexNet':
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    elif model_name.startswith('VGG16'):
        for c,nc in [(1,2), (2,2), (3,3), (4,3), (5,3)]:
            layers += ['conv%d_%d' % (c,i+1) for i in xrange(nc)]
        layers += ['fc6', 'fc7']
    elif model_name.startswith('VGG19'):
        for c,nc in [(1,2), (2,2), (3,4), (4,4), (5,4)]:
            layers += ['conv%d_%d' % (c,i+1) for i in xrange(nc)]
        layers += ['fc6', 'fc7']
    if not short:
        layers = [l + 'ex' for l in layers]
    return layers

def get_model_rf_sizes(model_name):
    if model_name == 'AlexNet':
        rf_sizes = [11, 51, 99, 131, 163, 227, 227]
    elif model_name.startswith('VGG16'):
        rf_sizes = [3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196, 224, 224]
    elif model_name.startswith('VGG19'):
        rf_sizes = [3, 5, 10, 14, 24, 32, 40, 48, 68, 84, 100, 116, 156, 188, 220, 224, 224, 224]
    else:
        rf_sizes = []
    return rf_sizes

def plot_rf_sizes(model_name, spec):
    import matplotlib.pyplot as plt
    import numpy as np
    rfs = get_model_rf_sizes(model_name)
    n = len(rfs)
    x = np.linspace(0, 1, n)
    plt.plot(x, rfs, spec)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot_rf_sizes('AlexNet', 'b-')
    plot_rf_sizes('VGG16', 'r-')
    plot_rf_sizes('VGG19', 'k-')
    plt.show()


########################################################################################################################
# Primate experiments

def get_experiment_db_filename(set_index, set_name='turk'):
    if set_name == 'turk':
        return os.path.join(experiment_path, 'set%d.db' % set_index)
    else:
        return os.path.join(experiment_path, '%s_%d.db' % (set_name, set_index))

def get_experiment_db_filename_by_run(experiment_run):
    return os.path.join(experiment_path, '%s.db' % (experiment_run))

def get_experiment_imageset(experiment_index):
    return experiment_index, 'turk'

