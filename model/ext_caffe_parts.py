#!/usr/bin/python
import lmdb
import numpy as np
import caffe
import os
import rapid_categorization.levels.util

def ext_caffe_part(lmdb_fn, out_fn, set_name, i_set):
    # Load set
    lbl, src_fns = rapid_categorization.levels.util.load_labels(set_name, i_set)
    n_samples = lbl.size
    out_data = None
    print '%d samples in %s.' % (n_samples, lmdb_fn)
    # Load LMDB
    with lmdb.open(lmdb_fn, readonly=True) as lmdb_env:
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        has_data = lmdb_cursor.first()
        for i in xrange(n_samples):
            if not (i % 100):
                if not (i % 1000):
                    print '%d' % i,
                else:
                    print '.',
            # Get current sample data
            if not has_data:
                raise RuntimeError('Unexpected end of LMDB.')
            (k,v) = lmdb_cursor.item()
            datum = caffe.io.string_to_datum(v)
            a = caffe.io.datum_to_array(datum)
            # Get current sample label
            #label = source_labels[i]
            # Create target set if not done already
            if out_data is None:
                out_data = np.zeros((n_samples, a.size))
            # Assign data to set
            out_data[i,:] = np.reshape(a, -1)
            # Next entry!
            has_data = lmdb_cursor.next()
        print 'done.'
    # Store resuls
    np.savez(out_fn, data=out_data)

def ext_caffe_parts(layer, set_name, path, min, max):
    for i in xrange(min, max):
        lmdb_fn = os.path.join(path, '%05d_%s.lmdb' % (i, layer))
        out_fn = os.path.join(path, '%s_%05d.npz' % (set_name, i))
        if not os.path.isfile(out_fn):
            ext_caffe_part(lmdb_fn, out_fn, set_name, i)
        else:
            print 'skipping %s' % out_fn

if __name__ == "__main__":
    ext_caffe_parts('fc7', 'setb50k', '/media/data_cifs/sven2/ccv/AnNonAn_cc256/caffe_fc7', 500, 501)

