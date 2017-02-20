#!/usr/bin/env python
# Per-run settings of the experiments

def base_settings(p):
    p['config'] = {}
    p['config']['HIT Configuration'] = {}
    p['exp'] = {}
    p['exp']['num_blocks'] = 6
    p['exp']['fixation_duration'] = 500
    p['exp']['presentation_duration'] = 100
    p['exp']['training_presentation_durations'] = [500, 500, 500, 500, 200, 200, 100, 100]
    p['exp']['max_answer_times'] = [500, 500, 500, 500, 500, 500]
    p['exp']['max_training_answer_time'] = 1500
    p['exp']['answer_keys'] = [83, 76]
    p['exp']['answers'] = ['S', 'L']
    p['exp']['answer_strings'] = ['animal', 'non-animal']
    p['exp']['example_stims'] = ["static/examples/ex1n.webm",
                          "static/examples/ex1a.webm",
                          "static/examples/ex2n.webm",
                          "static/examples/ex2a.webm",
                          "static/examples/ex3a.webm",
                          "static/examples/ex3n.webm"]
    p['exp']['example_stim_len'] = [1300, 1300, 1300, 1300, 1300, 1300]
    p['exp']['example_answers'] = [1, 0, 1, 0, 0, 1]
    p['exp']['example_pretime'] = 1100


def clicktionary(p):
    base_settings(p)
    p['video_base_path'] = '/media/data_clicktionary/rapid_categorization/clicktionary_masked_images_balanced_cut_100'
    p['example_path'] = '/media/data_clicktionary/rapid_categorization/masked_examples'
    p['set_name'] = 'clicktionary'
    p['set_indices'] = range(2000, 2020)
    p['config']['HIT Configuration']['title'] = 'Animal Realization'
    p['config']['HIT Configuration']['description'] = 'Categorize whether or not a scrambled image contains an animal'

def clicktionary50ms(p):
    base_settings(p)
    p['video_base_path'] = '/media/data_clicktionary/rapid_categorization/clicktionary_masked_images_balanced_cut_50'
    p['example_path'] = '/media/data_clicktionary/rapid_categorization/masked_examples'
    p['set_name'] = 'clicktionary'
    p['set_indices'] = range(2000, 2020)
    p['config']['HIT Configuration']['title'] = 'Animal Realization'
    p['config']['HIT Configuration']['description'] = 'Categorize whether or not a scrambled image contains an animal'
    p['exp']['presentation_duration'] = 50