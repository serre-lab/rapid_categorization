# Configuration settings per machine

# Public-facing server to run experiments on
deploy_machine = 'turk'
deploy_folder_name = 'psiturk_experiment'

# Folder to store data
video_path = '/media/data_clicktionary/rapid_categorization/'

# Particular experiment to run. Determines subfolders.
experiment_name = 'clicktionary'
onset_times_ms = [1000, 1100, 1200, 1300, 1400, 1500, 1600]
after_time_ms = 500
stim_show_time_ms = 50
input_image_path = '/media/data_cifs/clicktionary/causal_experiment/clicktionary_masked_images'

# Folder to store experiment runs
psiturk_run_path = '/home/sven2/psiturk/runs'