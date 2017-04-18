# Configuration settings per machine

import platform

# Public-facing server to run experiments on
deploy_machine = 'turk'
deploy_folder_name = '/home/psiturkey/experiments'

# Folder to store data
video_path = '/media/data_clicktionary/rapid_categorization/'

# Particular experiment to run. Determines subfolders.

# Local folder to store experiment runs
hostname = platform.node()
if hostname == 'x8':
    psiturk_run_path = '/home/sven2/psiturk/runs'
elif hostname == 'g17':
    psiturk_run_path = '/home/charlie/psiturk/runs'
elif hostname == 'x9':
    psiturk_run_path = '/home/drew/Documents/psiturk/runs'
else:
    raise RuntimeError('Add your path to the config.')
