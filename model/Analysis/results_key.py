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
    myFile = open(img_list,'r')
    for line in myFile:
        im = line.split()[0]
        lab = line.split()[1]
        im = im+'.jpg'
        #print im
        if lab == '1':
            label = 'animal'
        else:
            label = 'nonanimal'
        im2label[im] = label
    myFile.close()
    return im2label
