#!/usr/bin/env python
# Configuration for clicktionary

imageset_base_path = '/media/data_cifs/clicktionary/causal_experiment'
plot_path = '/media/data_cifs/clicktionary/causal_experiment'
pickle_path = '/media/data_cifs/clicktionary/causal_experiment'

def get_experiment_sets(name):
    if name == 'clicktionary': return 50, 'clicktionary'
    if name == 'clicktionary50ms': return 70, 'clicktionary'
    if name == 'clicktionary50msfull': return 70, 'clicktionary'
