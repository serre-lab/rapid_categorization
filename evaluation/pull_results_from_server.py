#!/usr/bin/env python
# Copy categorization DB from g15 to CIFS

import os, subprocess
from hmax.levels import util
from rapid_categorization import config

def pull_results(experiment_run):
    cifs_filename = util.get_experiment_db_filename_by_run(experiment_run)
    remote_filename = config.deploy_machine + ':' + os.path.join(config.deploy_folder_name, experiment_run, 'participants.db')
    if not os.path.exists(cifs_filename):
        subprocess.call(['scp', remote_filename, cifs_filename])
        print 'File saved to %s' % cifs_filename
    else:
        print 'File exists: %s' % cifs_filename


if __name__ == '__main__':
    pull_results('clicktionary400msvaranswerfull')