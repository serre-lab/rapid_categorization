# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:41:31 2016

@author: jcader
"""

def label_results(img_list):
    """
    returns a dictionary mapping images to ground truth 
    """
    im2label = {}
    num_animal = 0
    num_nonanimal = 0
    for line in open(img_list,'r').read().splitlines():
        im, lab = line.split('\t')
        if lab == '1':
            label = 'animal'
            num_animal += 1
        else:
            label = 'nonanimal'
            num_nonanimal += 1
        im2label[im] = label
    assert(num_nonanimal and num_animal)
    return im2label
