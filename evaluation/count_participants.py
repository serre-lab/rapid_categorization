#!/usr/bin/env python
# Count participants over subset of experiments

import os, glob
from rapid_categorization.model.util import experiment_path
from rapid_categorization.evaluation.data_loader import Data

def print_participant_count(experiment_wildcard):
    # Get participants from all experiments matching wildcard
    exp_fns = glob.glob(os.path.join(experiment_path, experiment_wildcard + '.db'))
    all_ids = []
    unique_ids = set()
    n_experiments_with_subjects = 0
    for exp_fn in exp_fns:
        exp = os.path.splitext(os.path.basename(exp_fn))[0]
        data = Data()
        data.get_participant_ids(exp)
        ids = data.workerIds
        if len(ids):
            all_ids += ids
            for id in ids:
                unique_ids.add(id)
            n_experiments_with_subjects += 1
    print 'Rapid MTurk for wildcard %s:' % experiment_wildcard
    print '  %d experiments' % n_experiments_with_subjects
    print '  %d participants' % len(all_ids)
    print '  %d unique participants' % len(unique_ids)
    if len(unique_ids):
        print 'Duplicates ids: %s' % [item for item in set(all_ids) if all_ids.count(item) > 1]


if __name__ == '__main__':
    print_participant_count('click_center_probfill_400stim_150res_combined')
