#!/usr/bin/python

import caffe
import numpy as np
import random
import os
import sys

subset_filenames = '/media/sven2/data/sven2/AnNonAn/subset_%s_%d_%d.npy'
subset_num = 10000

# TODO: Generalize for more layers

class ext_subset_conv3(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one input.")
        self.ctx = None

    def reshape(self, bottom, top):
        # Generate subset or loaded it if already present
        self.n_batch = bottom[0].data.shape[0]
        self.n_inputs = bottom[0].data.size / self.n_batch
        subset_filename = subset_filenames % ('conv3', subset_num, self.n_inputs)
        if os.path.isfile(subset_filename):
            sys.stderr.write('Feature subset: Load extract file %s.\n' % subset_filename)
            self.subset = np.load(subset_filename)
        else:
            sys.stderr.write('Feature subset: Create extract file %s.\n' % subset_filename)
            self.subset = random.sample(xrange(self.n_inputs), subset_num)
            np.save(subset_filename, self.subset)
        # top shape is reduced and flattened
        top[0].reshape(self.n_batch, subset_num, 1, 1)
        self.output = np.zeros((self.n_batch, subset_num, 1, 1))

    def forward(self, bottom, top):
        #sys.stderr.write('Shapes are %s %s %s\n' % (str(top[0].data.shape), str(bottom[0].data.shape), str(self.output.shape)))
        for i in xrange(self.n_batch):
            input = bottom[0].data[i].flatten()
            sub_input = input[self.subset]
            self.output[i,:,0,0] = sub_input
        top[0].data[...] = self.output
        sys.stderr.write('FORWARD!!!\n')

    def backward(self, top, propagate_down, bottom):
        raise RuntimeError('Nono backprop on extraction.')

    def __del__(self):
        pass