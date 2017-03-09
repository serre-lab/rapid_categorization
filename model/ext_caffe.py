#!/usr/bin/python
import lmdb
import numpy as np
import caffe

in_fn = '/media/sven2/data/sven2/sets/all_fc7.lmdb'
in_labels = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/sets/all.txt'
out_set = 'setb50k'
out_set_count = 919
out_fns = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/caffe_fc7/setb50k_%05d.npz'
n_convert = 1410000
batch_size = 5000

# Get association between caffe training set (in_labels) and desired output subsets (out_set)
with open(in_labels, 'rt') as fid:
    source_labels = fid.read().splitlines()

out_set_map = dict()
for i_set in xrange(out_set_count):
    set_fn = '/media/clpsshare/sven2/ccv/AnNonAn_cc256/sets/%s_%05d.txt' % (out_set, i_set)
    if i_set % 100 == 0:
        print 'Loading set %s...' % set_fn
    with open(set_fn, 'rt') as fid:
        out_set_labels = fid.read().splitlines()
    for i_set_idx,label in enumerate(out_set_labels):
        out_set_map[label] = (i_set,i_set_idx)

# Prepare output arrays
out_data = [None] * out_set_count
out_count = [0] * out_set_count

# Collect data from LMDB
with lmdb.open(in_fn, readonly=True) as lmdb_env:
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    has_data = lmdb_cursor.first()
    for i in xrange(n_convert):
        # Get current sample data
        if not has_data:
            print 'Unexpected end of LMDB.'
            break
        (k,v) = lmdb_cursor.item()
        datum = caffe.io.string_to_datum(v)
        a = caffe.io.datum_to_array(datum)
        # Get current sample label
        label = source_labels[i]
        # Find set and position in set this sample belongs to
        target_set = out_set_map.get(label)
        if target_set is None:
            print 'Entry [%s] not found in target sets. Skipping.' % label
            continue
        i_set = target_set[0]
        i_setidx = target_set[1]
        if i % 1000 == 0:
            print 'Entry %d [%s] in set %s.' % (i, label, str(target_set))
        # Create target set if not done already
        if out_data[i_set] is None:
            out_data[i_set] = np.zeros((batch_size, a.size))
        # Assign data to set
        out_data[i_set][i_setidx,:] = np.reshape(a, -1)
        out_count[i_set] += 1
        # Set completed?
        if out_count[i_set] == batch_size:
            print 'SET DONE! %04d' % (i_set)
            out_fn = out_fns % i_set
            np.savez(out_fn, data=out_data[i_set])
        # Next entry!
        has_data = lmdb_cursor.next()

# Save data
for i_set in xrange(out_set_count):
    if out_count[i_set] != batch_size:
        print 'WARNING: Set %d missing %d images!' % (i_set, batch_size - out_count[i_set])
        #out_data[i_set] = out_data[i_set][:out_count[i_set]]
    else:
        # Already saved
        continue
    out_fn = out_fns % i_set
    np.savez(out_fn, data=out_data[i_set])
