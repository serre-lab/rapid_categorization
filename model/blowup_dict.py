#!/usr/bin/env python
# take S2 dictionary in list-of-entries format and convert it to 4D array with zero-padding

import os
import pickle
import numpy as np

max_feature_size = 12

base_path = '/users/seberhar/data/data/AnNonAn_cc256' # CCV
if not os.path.exists(base_path):
    base_path = '/home/sven2/DeleteMe/convtest' # local test
if not os.path.exists(base_path):
    base_path = '/media/data_cifs/sven2/ccv/' # local test


dict_in = os.path.join(base_path, 'hmax_c1', 'dict.pickle')
dict_out = os.path.join(base_path, 'hmax_c1', 'dict4d.pickle')

# Load
with open(dict_in, 'rb') as fid:
    old_dict = pickle.load(fid)


# Convert
n = len(old_dict)
assert(n > 0)
new_dict_size = 16
new_dict = np.zeros((n, old_dict[0].shape[0], new_dict_size, new_dict_size), np.float32)
sizes = [[0]] * n
j = 0
for i,f in enumerate(old_dict):
    feature_size = f.shape[-1]
    if feature_size > max_feature_size:
        continue
    new_dict[j, :, 0:f.shape[-2], 0:f.shape[-1]] = f
    sizes[j] = [feature_size]
    j += 1

print 'Converted %d features. Skipped %d oversized.' % (j, n - j)
n = j

d = dict()
d['n_words'] = n
d['l_words'] = new_dict[:n]
d['w_sizes'] = sizes[:n]

# Save
with open(dict_out, 'wb') as fid:
    pickle.dump(d, fid)
