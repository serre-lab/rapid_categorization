import numpy as np
import cPickle
import os
import re


def convert_dict(data_dict):
    out_dict = {}
    out_dict['revelation_raw'] = [200 if x.split(
        '/')[-1].split('_')[0] == 'full' else int(
            x.split('/')[-1].split('_')[0]) for x in np.concatenate(
            data_dict['files'])]
    out_dict['true_labels'] = np.concatenate(data_dict['labs'])
    if len(data_dict['preds'].shape) > 1:
        preds = np.concatenate(data_dict['preds'][:])
    else:
        preds = data_dict['preds']
    out_dict['correctness_raw'] = preds == out_dict['true_labels']
    out_dict['source_filenames'] = ['%s/%s' % (
        x.split('/')[-1].split('_')[0], re.split(
            '(\d+_)', x.split('/')[-1])[-1].split(
            '.png')[0]) for x in np.concatenate(data_dict['files'])]
    return out_dict


def save_pickle(out_pointer, data):
    with open(out_pointer, 'wb') as fid:
        cPickle.dump(data, fid)


def main(
    model_dir='/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints/',
    model_name='clickme_vgg16_fc7_01_2017_05_17_11_56_08',  # 'vgg16_fc7_01_2017_05_15_22_53_47',  # 'vgg16_fc7_01_2017_05_14_12_02_05',
    out_dir='/media/data_cifs/clicktionary/causal_experiment',
    out_file='perf_by_revelation_clicktionary_%s.p',
    set_name=2020):  # 1000s are main clickme model results. + 1000 is the synthetics

    main_results = os.path.join(
        model_dir, model_name, 'validation_results.npz')
    main = convert_dict(np.load(main_results))
    out_pointer = os.path.join(out_dir, out_file % set_name)
    save_pickle(out_pointer, main)

    try:
        synthetic_subject_results = os.path.join(
            model_dir, model_name, 'validation_results_sim_subs.npy')
        synthetic_subjects = [convert_dict(x) for x in np.load(
            synthetic_subject_results)]
        out_pointer = os.path.join(out_dir, out_file % (set_name + 1000))
        save_pickle(out_pointer, synthetic_subjects)
    except:
        'Failed to save synthetic subjects.'


if __name__ == '__main__':
    main()
