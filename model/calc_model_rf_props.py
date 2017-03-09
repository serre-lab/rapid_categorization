#!/usr/bin/env python

# Determine model RF sizes and strides per layer

from google.protobuf import text_format
import numpy as np
import caffe

def print_net_rfs(net_fn):
    print '===== %s:' % net_fn
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    stride = 1
    rf_size = 1
    for layer in model.layer:
        if layer.type == "Convolution":
            p = layer.convolution_param
        elif layer.type == "Pooling":
            p = layer.pooling_param
        else:
            continue
        if isinstance(p.kernel_size, int):
            ks = p.kernel_size
        else:
            if len(p.kernel_size) >= 1:
                ks = p.kernel_size[0]
            else:
                ks = 1
        if isinstance(p.stride, int):
            s = p.stride
        else:
            if len(p.stride) >= 1:
                s = p.stride[0]
            else:
                s = 1
        rf_size = (ks-1)*stride + rf_size
        stride *= s
        if layer.type == "Convolution":
            print "%s: %d" % (layer.name, rf_size)

if __name__ == '__main__':
    print_net_rfs('deploy_fc7.prototxt')
    print_net_rfs('new_VGG16_fc7.prototxt')
    print_net_rfs('new_VGG19_fc7.prototxt')
