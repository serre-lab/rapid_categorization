#!/usr/bin/env python
# Take the base experiment from psiturk_experiment and create a copy in the runs/ folder
# with the appropriate settings

import os, shutil
from rapid_categorization.run_settings import settings
from hmax.levels.util import get_imageset_filename
from rapid_categorization.config import psiturk_run_path
import subprocess

def apply_dict_to_template(src_filename, dst_filename, settings):
    # Replace placeholders in src filename and save to dst filename
    data = open(src_filename, 'rt').read()
    for k, v in settings.iteritems():
        data = data.replace('{{%s}}' % k, str(v))
    open(dst_filename, 'wt').write(data)

def dict_to_js(settings, dst_filename):
    # Save all entries in settings dictionary to javascript
    # (Current supports only numbers, strings and arrays of supported types
    #  because they are compatible between python %s and js)
    with open(dst_filename, 'wt') as fid:
        for k,v in settings.iteritems():
            fid.write('var %s = %s;\n' % (k, str(v)))

def write_set_file(set_index, set_name, dst_filename):
    # Load set and convert to CSV
    src_fn = get_imageset_filename(set_index=set_index, set_name=set_name)
    data = open(src_fn, 'rt').read().splitlines()
    with open(dst_filename, 'wt') as fid:
        fid.write('img,cat\n')
        for line in data:
            fid.write(','.join(line.split('\t')) + '\n')

def generate_experiment(name, force_overwrite=False, deploy=False):
    # Get settings
    p = {}
    getattr(settings, name)(p)
    # Determine source and target folders for experiment
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    base_path = os.path.join(root_path, 'psiturk_experiment')
    run_path = os.path.join(psiturk_run_path, name)
    # Already generated?
    if os.path.isdir(run_path):
        if not force_overwrite:
            raise RuntimeError('Experiment at %s already exists.' % run_path)
        shutil.rmtree(run_path)
    # Copy base experiment
    shutil.copytree(base_path, run_path, symlinks=True)
    # Link videos
    os.symlink(p['video_base_path'], os.path.join(run_path, 'static', 'dataset'))
    os.symlink(p['example_path'], os.path.join(run_path, 'static', 'examples'))
    # Write set files
    for set_index in p['set_indices']:
        write_set_file(set_index=set_index, set_name=p['set_name'], dst_filename=os.path.join(run_path, 'static', 'set_%d.csv' % set_index))
    # Set filenames to JS config
    p['exp']['experiment_sets'] = p['set_indices']
    # Generate javascript settings file
    dict_to_js(p['exp'], os.path.join(run_path, 'static', 'config.js'))
    # deploy to server
    if deploy:
        subprocess.call(['rsync', '-avz', '--', run_path, 'turk:~/experiments/'])

def sync_stimuli():
    subprocess.call(['rsync', '-aLvz', '--', '/media/data_clicktionary/rapid_categorization', 'turk:/media/data_clicktionary/'])

if __name__ == '__main__':
    generate_experiment('clicktionary', force_overwrite=True, deploy=True)
    sync_stimuli()
