# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:17:17 2015

@author: jcader
"""

myFile = open("setb50k_00500.txt")
ims = []
im_name = {}
animals = 0
nonanimals = 0
for line in myFile:
        ims.append(line)
myFile.close()

myFile = open('setb50k_00500_300.txt', 'w')
for i in range(300):
    myFile.write(str(ims[i]).strip()+'\n')
    line = ims[i].strip()
    if line.split()[0] in im_name.keys():
        im_name[line.split()[0]] +=1
    else:
        im_name[line.split()[0]] = 1
    if line.split()[1] == '0':
    	nonanimals +=1
    else:
    	animals +=1
myFile.close()

#Check everything worked
duplicates = False
for im in im_name.keys():
	if im_name[im] > 1:
		duplicates = True
if duplicates == False:
	print "All images are unique"
else:
	print "ERROR: duplicate image(s) in the test set"
print "Animal images: "+ str(animals), "Non-Animal images: " + str(nonanimals)
