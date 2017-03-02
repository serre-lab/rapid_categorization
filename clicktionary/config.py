#!/usr/bin/env python
# Configuration for clicktionary

imageset_base_path = '/media/data_cifs/clicktionary/causal_experiment'
plot_path = '/media/data_cifs/clicktionary/causal_experiment'
pickle_path = '/media/data_cifs/clicktionary/causal_experiment'

experiment_descs = {
    'clicktionary': 'Sets 1+2, 100ms stim, 500ms answer',
    'clicktionary50ms': 'Sets 1+2, 50ms stim, 500ms answer',
    'clicktionary50msfull': 'Set 1, Animal vs Vehicle, 50ms stim + 500ms answer',
    'clicktionary400msfull': 'Set 1, Animal vs Vehicle, 400ms stim + 500ms answer',
    'clicktionary400ms150msfull': 'Set 1, Animal vs Vehicle, 400ms stim + 150ms answer',
    'clicktionary400msvaranswerfull': 'Set 1, Animal vs Vehicle, 400ms stim + [100-500] ms answer'
}

def get_experiment_sets(name):
    if name == 'clicktionary': return 50, 'clicktionary'
    if name == 'clicktionary50ms': return 70, 'clicktionary'
    if name == 'clicktionary50msfull': return 70, 'clicktionary'
    if name == 'clicktionary400msfull': return 70, 'clicktionary'
    if name == 'clicktionary400ms150msfull': return 70, 'clicktionary'
    if name == 'clicktionary400msvaranswerfull': return 70, 'clicktionary'

def get_experiment_desc(name):
    return experiment_descs[name]