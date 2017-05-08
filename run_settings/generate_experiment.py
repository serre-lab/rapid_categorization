#!/usr/bin/env python
# Take the base experiment from psiturk_experiment and create a copy in the runs/ folder
# with the appropriate settings

import os, shutil
from rapid_categorization.run_settings import settings
from rapid_categorization.model.util import get_imageset_filename
from rapid_categorization.config import psiturk_run_path
import subprocess
from ConfigParser import SafeConfigParser
from rapid_categorization.prepare_stimuli.generate_stimulus_videos import generate_stimulus_videos
from rapid_categorization.evaluation import duplicates_data_loader

def apply_dict_to_template(src_filename, dst_filename, settings):
    # Replace placeholders in src filename and save to dst filename
    data = open(src_filename, 'rt').read()
    n_replaced = 0
    replaced_strings = set()
    for k, v in settings.iteritems():
        key_string = '{{%s}}' % k
        n = data.count(key_string)
        n_replaced += n
        if n: replaced_strings.add(k)
        data = data.replace(key_string, str(v))
    open(dst_filename, 'wt').write(data)
    if n_replaced:
        print 'Replaced %d strings %s in %s' % (n_replaced, str(replaced_strings), dst_filename)

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
    class_identifiers = ['nonanimal', 'animal']
    with open(dst_filename, 'wt') as fid:
        fid.write('img,cat\n')
        for line in data:
            ll = line.split('\t')
            fn = ll[0]
            class_identifier = class_identifiers[int(ll[1])]
            fid.write('%s,%s\n' % (fn, class_identifier))

def write_values_to_config(settings, config_filename):
    cp = SafeConfigParser()
    cp.read(config_filename)
    for sect_name, sect_values in settings.iteritems():
        for k,v in sect_values.iteritems():
            cp.set(sect_name, k, str(v))
    cp.write(open(config_filename, 'wt'))

def apply_dict_to_tree(root_path, extensions, settings):
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                fn_full = os.path.join(subdir, file)
                apply_dict_to_template(fn_full, fn_full, settings)

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
    # Generate image sets
    video_base_path = p['video_base_path']
    if not os.path.isdir(video_base_path):
        print 'Did not found videos at %s' % video_base_path
        print 'Generating...'
        onset_times_ms = p['exp']['trial_pretimes']
        after_time_ms = max(p['exp']['max_answer_times'])
        input_image_path = p['input_image_path']
        stim_show_time_ms = p['exp']['presentation_duration']
        generate_stimulus_videos(input_image_path, video_base_path, onset_times_ms, after_time_ms, stim_show_time_ms=stim_show_time_ms)
    else:
        print 'Using existing videos at %s' % video_base_path
    # Copy base experiment
    shutil.copytree(base_path, run_path, symlinks=True)
    # Apply string replacements
    apply_dict_to_tree(run_path, ['.html', '.js'], p['identifiers'])
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
    # Generate main psiturk config file
    write_values_to_config(p['config'], os.path.join(run_path, 'config.txt'))
    # Create database with duplicate participants blocked (if requested)
    duplicates_data_loader.create_dummy_participant_db(
        experiment_names=p['exclude_participants'],
        output_filename= os.path.join(run_path, 'participants.db'))
    # deploy to server
    if deploy:
        subprocess.call(['rsync', '-avz', '--', run_path, 'turk:~/experiments/'])

def sync_stimuli():
    subprocess.call(['rsync', '-aLvz', '--', '/media/data_clicktionary/rapid_categorization', 'turk:/media/data_clicktionary/'])

if __name__ == '__main__':
    # generate_experiment('click_center_probfill_400stim_150res_5', force_overwrite=True, deploy=True)
    generate_experiment('artifact_vehicles_turk', force_overwrite=True, deploy=True)
    # sync_stimuli()
