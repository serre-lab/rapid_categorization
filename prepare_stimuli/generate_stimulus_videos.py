#!/usr/bin/env python
# Generate the stimulus videos for all images of the current rapid categorization experiment

import os
from rapid_categorization import config
import subprocess

def generate_stimulus_videos(image_folder, video_folder, onset_times_ms, after_time_ms, stim_show_time_ms):
    # Determine conversion script name
    stim_sh = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stim2video.sh')
    fps = 1000 / stim_show_time_ms
    # Recursively walk over files
    for root, dirs, files in os.walk(image_folder):
        print 'Processing images in %s...' % root
        # Create target directory
        target_path = os.path.join(video_folder, os.path.relpath(root, image_folder))
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        # Convert all files
        for file in files:
            fn_full = os.path.join(root, file)
            (file_base, ext) = os.path.splitext(file)
            if ext.lower() != '.png':
                print 'Ignoring non-.png file ', fn_full
                continue
            target_path_for_file = os.path.join(target_path, file_base)
            print '%s -> %s...' % (fn_full, target_path_for_file)
            if not os.path.isdir(target_path_for_file):
                os.makedirs(target_path_for_file)
            for fix_ms in onset_times_ms:
                out_fn = os.path.join(target_path_for_file, str(fix_ms) + '.webm')
                # Skip if already converted
                if os.path.isfile(out_fn):
                    continue
                cmd = [stim_sh, fn_full, out_fn, str(fix_ms), str(after_time_ms), str(fps)]
                print cmd
                subprocess.call(cmd)
            # Also keep a copy of the image in the video folder
            # TODO

if __name__ == '__main__':
    #generate_stimulus_videos(config.input_image_path, config.video_path, config.onset_times_ms, config.after_time_ms, config.stim_show_time_ms)
    generate_stimulus_videos('/media/data_cifs/clicktionary/causal_experiment/clicktionary_masked_mircs', '/media/data_cifs/rapid_categorization/clicktionary_masked_mircs_500', config.onset_times_ms, config.after_time_ms, stim_show_time_ms=500)
    generate_stimulus_videos('/media/data_cifs/clicktionary/causal_experiment/clicktionary_masked_mircs',
                             '/media/data_cifs/rapid_categorization/clicktionary_masked_mircs_200',
                             config.onset_times_ms, config.after_time_ms, stim_show_time_ms=200)
    generate_stimulus_videos('/media/data_cifs/clicktionary/causal_experiment/clicktionary_masked_mircs',
                         '/media/data_cifs/rapid_categorization/clicktionary_masked_mircs_100',
                         config.onset_times_ms, config.after_time_ms, stim_show_time_ms=100)