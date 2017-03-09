# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:32:33 2016

@author: jcader
"""

import numpy as np
import xml.etree.ElementTree as ET
import os
import wget
import pickle
import random
import math
import datetime
from PIL import Image
import copy
import shutil

def get_synset_root(path_to_xml):
    """
    Returns the synset root of ImageNet 
    """
    root = ET.parse(path_to_xml).getroot()
    ImNetTree = root[1]
    print 'Using '+ ImNetTree.attrib['words']
    return ImNetTree

def make_wnimdMap(root):
    for elmt in root.iter():
        # animals
        if not(elmt.findall("[@wnid='n01905661']") == []): 
            invert = elmt.find("[@wnid='n01905661']")
        if not(elmt.findall("[@wnid='n01503061']") == []): 
            bird = elmt.find("[@wnid='n01503061']")
        if not(elmt.findall("[@wnid='n01627424']") == []): 
            amph = elmt.find("[@wnid='n01627424']")
        if not(elmt.findall("[@wnid='n02512053']") == []): 
            fish = elmt.find("[@wnid='n02512053']")
        if not(elmt.findall("[@wnid='n01661091']") == []): 
            rept = elmt.find("[@wnid='n01661091']")
        if not(elmt.findall("[@wnid='n01861778']") == []): 
            mamm = elmt.find("[@wnid='n01861778']")
        if not(elmt.findall("[@wnid='n02121808']") == []): 
            dcat = elmt.find("[@wnid='n02121808']")
        if not(elmt.findall("[@wnid='n02084071']") == []): 
            ddog = elmt.find("[@wnid='n02084071']")
        # non-animals
        if not(elmt.findall("[@wnid='n04341686']") == []): 
            struct = elmt.find("[@wnid='n04341686']")
        if not(elmt.findall("[@wnid='n03575240']") == []): 
            instr = elmt.find("[@wnid='n03575240']")
        if not(elmt.findall("[@wnid='n03093574']") == []): 
            consgood = elmt.find("[@wnid='n03093574']")
        if not(elmt.findall("[@wnid='n00017222']") == []): 
            plant = elmt.find("[@wnid='n00017222']")
        if not(elmt.findall("[@wnid='n09287968']") == []): 
            geo = elmt.find("[@wnid='n09287968']")
        if not(elmt.findall("[@wnid='n00019128']") == []): 
            natobj = elmt.find("[@wnid='n00019128']")
    #compile a list of all leaf nodes in each class
            # structure instrument, consumer goods
    sampling_dict = {}
    sampling_dict['mammals'] = {}
    sampling_dict['nonmammals'] = {}
    sampling_dict['natural'] = {}
    sampling_dict['artificial'] = {}
    
    rep_dict = {}
    rep_dict['mammals'] = {}
    rep_dict['nonmammals'] = {}
    rep_dict['natural'] = {}
    rep_dict['artificial'] = {}
    
    
    #create mapping to categories
    mmls = [mamm,dcat,ddog]
    nmmls = [invert,bird,amph,fish,rept]
    nat = [plant,geo,natobj]
    art = [struct,instr,consgood]
    wnidMap = {}
    wnidMap['mammals'] = []
    wnidMap['nonmammals'] = []
    wnidMap['natural'] = []
    wnidMap['artificial'] = []
    for cat in mmls:
        for wnid in cat.iter():
            wnidMap['mammals'].append(wnid.attrib['wnid'])
    for cat in nmmls:
        for wnid in cat.iter():
            wnidMap['nonmammals'].append(wnid.attrib['wnid'])
    for cat in nat:
        for wnid in cat.iter():
            wnidMap['natural'].append(wnid.attrib['wnid'])
    for cat in art:
        for wnid in cat.iter():
            wnidMap['artificial'].append(wnid.attrib['wnid'])
    return wnidMap


def wnid2cat(wonid,wnidMap):
    """ Maps WordNet IDs to their appropriate category for sampling
    """

    if wonid in wnidMap['mammals']:
        return 'mammals'
    elif wonid in wnidMap['nonmammals']:
        return 'nonmammals'
    elif wonid in wnidMap['natural']:
        return 'natural'
    else:
        return 'artificial'

def download_rep_ims(root,base_outdir,labels_path,shouldGen_imDict,shouldGen_sampleDict,setid=None):
    """ Creates and populates file structure to hold replacement images (25% total)
        from which annotators can pull if neccessary. Also parses ImageNet tree
        to create dictionaries of relevant entries for sampling
    """
    #find class roots
    for elmt in root.iter():
        # animals
        if not(elmt.findall("[@wnid='n01905661']") == []): 
            invert = elmt.find("[@wnid='n01905661']")
        if not(elmt.findall("[@wnid='n01503061']") == []): 
            bird = elmt.find("[@wnid='n01503061']")
        if not(elmt.findall("[@wnid='n01627424']") == []): 
            amph = elmt.find("[@wnid='n01627424']")
        if not(elmt.findall("[@wnid='n02512053']") == []): 
            fish = elmt.find("[@wnid='n02512053']")
        if not(elmt.findall("[@wnid='n01661091']") == []): 
            rept = elmt.find("[@wnid='n01661091']")
        if not(elmt.findall("[@wnid='n01861778']") == []): 
            mamm = elmt.find("[@wnid='n01861778']")
        if not(elmt.findall("[@wnid='n02121808']") == []): 
            dcat = elmt.find("[@wnid='n02121808']")
        if not(elmt.findall("[@wnid='n02084071']") == []): 
            ddog = elmt.find("[@wnid='n02084071']")
        # non-animals
        if not(elmt.findall("[@wnid='n04341686']") == []): 
            struct = elmt.find("[@wnid='n04341686']")
        if not(elmt.findall("[@wnid='n03575240']") == []): 
            instr = elmt.find("[@wnid='n03575240']")
        if not(elmt.findall("[@wnid='n03093574']") == []): 
            consgood = elmt.find("[@wnid='n03093574']")
        if not(elmt.findall("[@wnid='n00017222']") == []): 
            plant = elmt.find("[@wnid='n00017222']")
        if not(elmt.findall("[@wnid='n09287968']") == []): 
            geo = elmt.find("[@wnid='n09287968']")
        if not(elmt.findall("[@wnid='n00019128']") == []): 
            natobj = elmt.find("[@wnid='n00019128']")
    #compile a list of all leaf nodes in each class
            # structure instrument, consumer goods
    animals = [invert,bird,amph,fish,rept,mamm,dcat,ddog]
    nonanimals = [struct,instr,consgood,plant,geo,natobj]
    an_leaves = []
    nan_leaves = []
    an_ims = {}
    nan_ims = {}
    sampling_dict = {}
    sampling_dict['mammals'] = {}
    sampling_dict['nonmammals'] = {}
    sampling_dict['natural'] = {}
    sampling_dict['artificial'] = {}
    
    rep_dict = {}
    rep_dict['mammals'] = {}
    rep_dict['nonmammals'] = {}
    rep_dict['natural'] = {}
    rep_dict['artificial'] = {}
    
    
    #create mapping to categories
    mmls = [mamm,dcat,ddog]
    nmmls = [invert,bird,amph,fish,rept]
    nat = [plant,geo,natobj]
    art = [struct,instr,consgood]
    wnidMap = {}
    wnidMap['mammals'] = []
    wnidMap['nonmammals'] = []
    wnidMap['natural'] = []
    wnidMap['artificial'] = []
    for cat in mmls:
        for wnid in cat.iter():
            wnidMap['mammals'].append(wnid.attrib['wnid'])
    for cat in nmmls:
        for wnid in cat.iter():
            wnidMap['nonmammals'].append(wnid.attrib['wnid'])
    for cat in nat:
        for wnid in cat.iter():
            wnidMap['natural'].append(wnid.attrib['wnid'])
    for cat in art:
        for wnid in cat.iter():
            wnidMap['artificial'].append(wnid.attrib['wnid'])

    for an in animals:
        for child in an.iter():
            if len(child) == 0:
                an_leaves.append(child.attrib['wnid'])
    for nan in nonanimals:
        for child in nan.iter():
            if len(child) == 0:
                nan_leaves.append(child.attrib['wnid'])
    
    #read in relevant image links
    if shouldGen_imDict:
        f = open(labels_path)
        tot = 0
        for line in f:
            tot += 1
        curr = -1
        f.close()
        
        f = open(labels_path)
        for line in f:
            curr +=1
            im_id = line.split()[0]
            url = line.split()[1]
            wnid = im_id.split('_')[0]
            if wnid in an_leaves:
                if not (wnid in an_ims.keys()):
                    an_ims[wnid] = {}
                    an_ims[wnid]['count'] = 0
                    an_ims[wnid]['ims'] = {}
                an_ims[wnid]['count'] += 1
                an_ims[wnid]['ims'][im_id] = url
            if wnid in nan_leaves:
                if not (wnid in nan_ims.keys()):
                    nan_ims[wnid] = {}
                    nan_ims[wnid]['count'] = 0
                    nan_ims[wnid]['ims'] = {}
                nan_ims[wnid]['count'] += 1
                nan_ims[wnid]['ims'][im_id] = url
            if curr%1000000 == 0:
                print str(curr)+' of '+str(tot)+' images parsed' 
        f.close()
        pickle.dump(an_ims,open("animals.p","wb"))
        pickle.dump(nan_ims,open("nonanimals.p","wb"))
    else:
        an_ims = pickle.load(open("animals.p","rb"))
        nan_ims = pickle.load(open("nonanimals.p","rb"))

    #split into sampling and replacement images
    if shouldGen_sampleDict:
        #for animals
        tot = len(an_ims.keys()) + len(nan_ims.keys())
        done = 0
        for wnid in an_ims.keys():
            for_rep = int(math.floor(an_ims[wnid]['count']/10)) #set aside 10% of images for manual replacement
            ims = an_ims[wnid]['ims'].keys()
            to_download = random.sample(ims,for_rep)
            for im in ims:
                cat = wnid2cat(wnid,wnidMap)
                if im in to_download:
                    if not (wnid in rep_dict[cat].keys()):
                        rep_dict[cat][wnid] = {}
                    rep_dict[cat][wnid][im] = an_ims[wnid]['ims'][im]
                
                else:
                    if not (wnid in sampling_dict[cat].keys()):
                        sampling_dict[cat][wnid] = {}
                    sampling_dict[cat][wnid][im] = an_ims[wnid]['ims'][im]
            if done%500 == 0:
                print str(done)+' of '+str(tot)+' categories split' 
            done +=1
            
        #for nonanimals               
        for wnid in nan_ims.keys():
            for_rep = int(math.floor(nan_ims[wnid]['count']/10)) #set aside 10% of images for manual replacement
            ims = nan_ims[wnid]['ims'].keys()
            to_download = random.sample(ims,for_rep)
            for im in ims:
                cat = wnid2cat(wnid,wnidMap)
                if im in to_download:
                    if not (wnid in rep_dict[cat].keys()):
                        rep_dict[cat][wnid] = {}
                    rep_dict[cat][wnid][im] = nan_ims[wnid]['ims'][im]
                else:
                    if not (wnid in sampling_dict[cat].keys()):
                        sampling_dict[cat][wnid] = {}
                    sampling_dict[cat][wnid][im] = nan_ims[wnid]['ims'][im]
            if done%500 == 0:
                print str(done)+' of '+str(tot)+' categories split' 
            done +=1
        setid = datetime.datetime.now().strftime("%y%m%d%H%M")
        pickle.dump(sampling_dict,open("sampling_dict"+setid+".p","wb"))
        pickle.dump(rep_dict,open("rep_dict"+setid+".p","wb"))
        
        
    base = base_outdir
    rep_dict = pickle.load(open("rep_dict"+setid+".p","rb"))
    print "Loaded image set "+str(setid)

    #for animals
    for wnid in an_ims.keys():
        cat = wnid2cat(wnid,wnidMap)
        if wnid in rep_dict[cat].keys():
            out_dir = os.path.join(base,'raw_ims','animal_extra',wnid)
            #ensure folder present
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ims = rep_dict[cat][wnid].keys()
            for im in ims:
                #download
                try:
                    os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+rep_dict[cat][wnid][im])
                except:
                    print "failed to download "+im
    
     #for nonanimals
    for wnid in nan_ims.keys():
        cat = wnid2cat(wnid,wnidMap)
        if wnid in rep_dict[cat].keys():
            out_dir = os.path.join(base,'raw_ims','nonanimal_extra',wnid)
            #ensure folder present
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            ims = rep_dict[cat][wnid].keys()
            for im in ims:
                #download
                try:
                    os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+rep_dict[cat][wnid][im])
                except:
                    print "failed to download "+im
    


def sample_set(n_outSets,n_ims,setid,sNum,base,wnidMap):
    sampling_dict = pickle.load(open("sampling_dict"+setid+".p","rb"))
    startNum = sNum
    if os.path.isfile('/home/jcader/Thesis/emtcat/sampled_list'+setid+'.p'):
        sampled_list = pickle.load(open('sampled_list'+setid+'.p','rb'))
    else:
        sampled_list = []
    n_an = n_ims/2
    n_nan = n_ims/2
    for outSet in range(n_outSets):
        print 'Starting to sample set '+str(outSet)
        out_dir = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(startNum))
        while os.path.exists(out_dir):
            startNum+=1
            out_dir = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(startNum))
        os.makedirs(out_dir)
        setName ='set'+setid+'_'+str(startNum)
        startNum += 1
        setlist = {}
        
        #animal - mammals
        mamms = []
        curr_dict = sampling_dict['mammals']
        ents = curr_dict.keys()
        print 'Sampling mammals'
        while len(mamms) < n_an/2:
            seed = random.randint(0,len(ents)-1)
            wnid = ents[seed]
            ims = curr_dict[wnid].keys()
            random.shuffle(ims)
            found = False
            curr = 0
            while ((found==False) and (curr < len(ims))):
                im = ims[curr]
                if not(im in sampled_list):
                    url = curr_dict[wnid][ims[curr]]
                    try:
                        os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+url)
                        found = True
                        mamms.append(im)
                        setlist[im] = 1
                        sampled_list.append(im)
                    except:
                        print "failed to download "+im
                        sampled_list.append(im) #note broken link
                        curr +=1
                else:
                    curr +=1
        #animal - nonmammals
        nonmamms = []
        curr_dict = sampling_dict['nonmammals']
        ents = curr_dict.keys()
        print 'Sampling non-mammal animals'
        while len(nonmamms) < n_an/2:
            seed = random.randint(0,len(ents)-1)
            wnid = ents[seed]
            ims = curr_dict[wnid].keys()
            random.shuffle(ims)
            found = False
            curr = 0
            while ((found==False) and (curr < len(ims))):
                im = ims[curr]
                if not(im in sampled_list):
                    url = curr_dict[wnid][ims[curr]]
                    try:
                        os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+url)
                        found = True
                        nonmamms.append(im)
                        setlist[im] = 1
                        sampled_list.append(im)
                    except:
                        print "failed to download "+im
                        sampled_list.append(im) #note broken link
                        curr +=1
                else:
                    curr +=1
                    
        #nonanimal - natural
        natural = []
        curr_dict = sampling_dict['natural']
        ents = curr_dict.keys()
        print 'Sampling natural non-animals'
        while len(natural) < n_nan/2:
            seed = random.randint(0,len(ents)-1)
            wnid = ents[seed]
            ims = curr_dict[wnid].keys()
            random.shuffle(ims)
            found = False
            curr = 0
            while ((found==False) and (curr < len(ims))):
                im = ims[curr]
                if not(im in sampled_list):
                    url = curr_dict[wnid][ims[curr]]
                    try:
                        os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+url)
                        found = True
                        natural.append(im)
                        setlist[im] = 0
                        sampled_list.append(im)
                    except:
                        print "failed to download "+im
                        sampled_list.append(im) #note broken link
                        curr +=1
                else:
                    curr +=1    
    
        #animal - mammals
        artificial = []
        curr_dict = sampling_dict['artificial']
        ents = curr_dict.keys()
        print 'Sampling artifical non-animals'
        while len(artificial) < n_nan/2:
            seed = random.randint(0,len(ents)-1)
            wnid = ents[seed]
            ims = curr_dict[wnid].keys()
            random.shuffle(ims)
            found = False
            curr = 0
            while ((found==False) and (curr < len(ims))):
                im = ims[curr]
                if not(im in sampled_list):
                    url = curr_dict[wnid][ims[curr]]
                    try:
                        os.system('wget --tries=1 -O '+os.path.join(out_dir,im)+' '+url)
                        found = True
                        artificial.append(im)
                        setlist[im] = 0
                        sampled_list.append(im)
                    except:
                        print "failed to download "+im
                        sampled_list.append(im) #note broken link
                        curr +=1
                else:
                    curr +=1
        
        # Update list of sampled images 
        pickle.dump(sampled_list,open('sampled_list'+setid+'.p','wb'))
        # write set text file
        ims = setlist.keys()
        
        random.shuffle(ims)
        f = open(os.path.join(base,'raw_ims','sets',setName+'.txt'),'w')
        for im in range(len(ims)):
            f.write(ims[im]+'\t'+str(setlist[ims[im]])+'\n')
        f.close()
                
            
def verify(setid,setnum,tot_im_num,base,wnidMap,useExt=False):
    outdir = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum))
    prefix = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum))
    sampling_dict = pickle.load(open("sampling_dict"+setid+".p","rb"))
    sampled_list = pickle.load(open('sampled_list'+setid+'.p','rb'))
    to_write = {}
    counts = {}
    counts['mammals'] = 0
    counts['nonmammals'] = 0
    counts['natural'] = 0
    counts['artificial'] = 0
    set_info = {}    
    # read in list of images
    f = open(os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum)+'.txt'),'r')
    for line in f:
        im_id = line.split()[0]
        wnid = im_id.split('_')[0]
        cat = wnid2cat(wnid,wnidMap)
        if ((cat == 'natural') or (cat == 'artificial')):
            lab = '0'
        else:
            lab = '1'
        set_info[im_id] = ['wnid','cat','lab']
        set_info[im_id][0] = wnid
        set_info[im_id][1] = cat
        set_info[im_id][2] = lab
    f.close()
    replacing = False
    replace_info = {}
    done_replacing = False
    version = 0
    im_ok = 0
    replacements_info = {}
    while not(done_replacing):
        version += 1
        print "Pass "+str(version)
        print len(to_write.keys())
        outdir_new = prefix+'_v'+str(version)
        os.mkdir(outdir_new)
        wnids_seen = []
        if replacing:
            set_info_old = copy.deepcopy(set_info)
            set_info = copy.deepcopy(replacements_info)
            replace_info = {}
            for im in to_write.keys():
                wnids_seen.append(to_write[im][0])
                if not(os.path.exists(os.path.join(outdir,im))):
                    im = im+'.jpg'
                shutil.move(os.path.join(outdir,im),os.path.join(outdir_new,im))
                
        #verify image
        for im in set_info.keys():
            bad_image = False
            no_file = False

            #see if file exists
            if ((useExt == True) and (replacing == False)):
                im_path = os.path.join(outdir,im+'.jpg')
            else:
                im_path = os.path.join(outdir,im)
            should_remove = False
            if not(os.path.isfile(im_path)):
                should_remove = True
                no_file = True
            #see if it's an openable image file
            try:
                img = Image.open(im_path)
                width, height = img.size
                white_corners = 0
                im_mat = np.asarray(img.convert('L'))
                for x in [0,width-1]:
                    for y in [0,height-1]:
                        if im_mat[y,x] > 237:
                            white_corners+=1
                if white_corners == 4:
                    should_remove = True
                if width < 256:
                    should_remove = True
                if height < 256:
                    should_remove = True
            except:
                should_remove = True
                bad_image = True
            #see if it has all white background
#            if (bad_image == False and no_file == False):
#             #check if below minimum size
#                img = Image.open(im_path)
#                width, height = img.size
#                white_corners = 0
#                im_mat = np.asarray(img.convert('L'))
#                for x in [0,width-1]:
#                    for y in [0,height-1]:
#                        if im_mat[y,x] > 237:
#                            white_corners+=1
#                if white_corners == 4:
#                    should_remove = True
#                if width < 256:
#                    should_remove = True
#                if height < 256:
#                    should_remove = True

#            # check if wnid has already appeared to avoid sampling with replacement
            wnid = set_info[im][0]
            if wnid in wnids_seen:
                should_remove = True
            if not(should_remove):
                to_write[im] = set_info[im]
                wnids_seen.append(wnid)
                counts[set_info[im][1]] +=1
                im_ok +=1
                
                shutil.move(im_path,os.path.join(outdir_new,im))
            else:
                replace_info[im] = set_info[im]
#                if no_file == False:
#                    os.remove(im_path)
        print str(im_ok)+'/'+str(tot_im_num)+' Images verified'
        shutil.rmtree(outdir,ignore_errors=True)
            
        # replace images to be replaced
        outdir = outdir_new
        rep_count = 1
        if replace_info == {}:
            done_replacing = True
        else: 
            replacing = True
            replacements_info = {}
            for img in replace_info.keys():
                print "Replacing "+img+" ("+str(rep_count)+"/"+str(len(replace_info.keys()))+" Images)"
                rep_count +=1
                cat = replace_info[img][1]
                curr_dict = sampling_dict[cat]
                ents = curr_dict.keys()
                seed = random.randint(0,len(ents)-1)
                wnid = ents[seed]
                if not(wnid in wnids_seen):
                    ims = curr_dict[wnid].keys()
                    random.shuffle(ims)
                    found = False
                    curr = 0
                    while ((found==False) and (curr < len(ims))):
                        im = ims[curr]
                        if not(im in sampled_list):
                            url = curr_dict[wnid][ims[curr]]
                            try:
                                os.system('wget --tries=1 -O '+os.path.join(outdir,im)+' '+url)
                                found = True
                                sampled_list.append(im)
                                replacements_info[im] = replace_info[img]
                                replacements_info[im][0] = wnid
                                replacements_info[im][1] = cat
                            except:
                                sampled_list.append(im) #note broken link
                                curr +=1
                        else:
                            curr +=1 
            temp_counts = copy.deepcopy(counts)
            placeholders = []
            for rep in replacements_info.keys():
                temp_counts[replacements_info[rep][1]] +=1
            for categ in temp_counts.keys():
                if temp_counts[categ] < tot_im_num/4:
                    diff = (tot_im_num/4) - temp_counts[categ]
                    for i in range(diff):
                        placeholders.append(categ)
            for plac in range(len(placeholders)):
                cat = placeholders[plac]
                if ((cat == 'natural') or (cat == 'artificial')):
                    lab = '0'
                else:
                    lab = '1'
                replacements_info['placeholder_'+str(plac)] = ['placeholder',cat,lab]
            

    #save new info
    outdir_new = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum))
    os.mkdir(outdir_new)
    for im in to_write.keys():
        shutil.move(os.path.join(outdir,im),os.path.join(outdir_new,im))
        cat = wnid2cat(to_write[im][1],wnidMap)
        if ((cat == 'natural') or (cat == 'artificial')):
            lab = '0'
        else:
            lab = '1'
    to_write[im][2] = lab
    shutil.rmtree(outdir,ignore_errors=True)
    os.remove(os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum)+'.txt'))
    f = open(os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum)+'.txt'),'w')
    ims = to_write.keys()
    random.shuffle(ims)
    for im_num in range(len(ims)):
        f.write(ims[im_num]+'\t'+str(to_write[ims[im_num]][2])+'\n')
    f.close()
    #update sampled list
    pickle.dump(sampled_list,open('sampled_list'+setid+'.p','wb'))

def process_for_AMT(setid,setnum,base):
    outdir = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum))
    # read in list of images
    set_info = {}
    f = open(os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum)+'.txt'),'r')
    for line in f:
        im_id = line.split()[0]
        lab = str(line.split()[1])
        wnid = im_id.split('_')[0]
        cat = wnid2cat(wnid,wnidMap)
        set_info[im_id] = ['wnid','cat','lab']
        set_info[im_id][0] = wnid
        set_info[im_id][1] = cat
        set_info[im_id][2] = lab
    f.close()

    for im in set_info.keys():
        im_path = os.path.join(outdir,im)
        if os.path.exists(im_path):
            img = Image.open(im_path)
            width, height = img.size
            # center crop to square
            if not((width == 256) and (height == 256)):
                if width > height:
                    left = (width - height)/2
                    top = 0
                    right = left + height
                    bottom = top + height
                else:
                    left = 0 
                    top = (height - width)/2
                    right = left + width
                    bottom = top + width 
                img = img.crop((left, top, right, bottom))
                # resize
                img = img.resize((256,256),Image.ANTIALIAS)
            # save
            img.convert('RGB').save(im_path+'.jpg')
            os.remove(im_path)

def save_norm_for_AMT(setid,setnum,base):
    ims = {}
    avgs = []            
    outdir = os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum))
    outdir_new = os.path.join(base,'TURK_IMAGES','set'+setid+'_'+str(setnum))
    if not(os.path.exists(outdir_new)):
        os.makedirs(outdir_new)
        # read in list of images
    set_info = {}
    f = open(os.path.join(base,'raw_ims','sets','set'+setid+'_'+str(setnum)+'.txt'),'r')
    for line in f:
        im_id = line.split()[0]
        lab = str(line.split()[1])
        wnid = im_id.split('_')[0]
        cat = wnid2cat(wnid,wnidMap)
        set_info[im_id] = ['wnid','cat','lab']
        set_info[im_id][0] = wnid
        set_info[im_id][1] = cat
        set_info[im_id][2] = lab
    f.close()
        
    for im in set_info.keys():
        im_path = os.path.join(outdir,im+'.jpg')
        img = Image.open(im_path)
        width, height = img.size
        #convert to Grayscale
        im_mat = np.asarray(img.convert('L'))
        # save
        avg = np.mean(im_mat)
        avgs.append(avg)
        #subract mean
        im_mat = im_mat - avg
        ims[im] = im_mat
    mean_val = np.mean(avgs)
    #add global mean
    for im in ims.keys():
        im_mat = ims[im]
        im_mat = im_mat + mean_val
        im_tosave = Image.fromarray(im_mat).convert('L')
        im_tosave.save(os.path.join(outdir_new,im+'.jpg'))        

def make_new_train(wnidMap,setid,base_extra,base_down,im_list,outdir,tot_num, move_ims=False):
    if move_ims:
        #Read in names of test images
        test_ims = []
        f = open(im_list,'r')
        for line in f:
            im_id = line.split()[0]
            test_ims.append(im_id)
        f.close()
        
        #initialize counts data structure 
        ims_used = test_ims
        to_write = {}
        to_write['mammals'] = []
        to_write['nonmammals'] = []
        to_write['natural'] = []
        to_write['artificial'] = []
        n_good = 0
        #Make directory for verified train set
        if not(os.path.exists(outdir)):
            os.makedirs(outdir)
        if not(os.path.exists('/home/jcader/Thesis/emtcat/initial_ims_test.p')):
            # Create list of verified images in folder
            for im in os.listdir(base_down):
                im_fn = os.path.join(base_down,im)
                #verfiy
                isgood = True
                #see if file exists
                if not(os.path.isfile(im_fn)):
                    isgood = False
                #see if it's an openable image file
                try:
                    img = Image.open(im_fn)
                    width, height = img.size
                    white_corners = 0
                    im_mat = np.asarray(img.convert('L'))
                    for x in [0,width-1]:
                        for y in [0,height-1]:
                            if im_mat[y,x] > 237:
                                white_corners+=1
                    if white_corners == 4:
                        isgood = False
                    if width < 256:
                        isgood = False
                    if height < 256:
                        isgood = False
                except:
                    isgood = False
                if im in ims_used:
                    isgood = False
                if isgood:
                    shutil.move(im_fn,os.path.join(outdir,im))
                    ims_used.append(im)
                    n_good += 1
                    wnid = im.split('_')[0]
                    to_write[wnid2cat(wnid,wnidMap)].append(im)
                if n_good%1000 == 0:
                        print str(n_good)+' images verified'
            pickle.dump(to_write,open('initial_ims_test.p','wb'))
        else:
            to_write = pickle.load(open('initial_ims_test.p','rb'))
            
        if not(os.path.exists('/home/jcader/Thesis/emtcat/extra_ims.p')):    
            #Read in other files
            print "Current: %d mammals, %d non-mammals, %d artificial, %d natural" % (len(to_write['mammals']),len(to_write['nonmammals']), len(to_write['artificial']), len(to_write['natural']))
            to_sample = {}
            to_sample['mammals'] = {}
            to_sample['nonmammals'] = {}
            to_sample['natural'] = {}
            to_sample['artificial'] = {}
            n_read = 0 
            #Animal targets
            for fold in os.listdir(os.path.join(base_extra,'animal_extra')):
                for im in os.listdir(os.path.join(base_extra,'animal_extra',fold)):
                    im_fn = os.path.join(base_extra,'animal_extra',fold,im)
                    #verfiy
                    isgood = True
                    #see if file exists
                    if not(os.path.isfile(im_fn)):
                        isgood = False
                    #see if it's an openable image file
                    try:
                        img = Image.open(im_fn)
                        width, height = img.size
                        white_corners = 0
                        im_mat = np.asarray(img.convert('L'))
                        for x in [0,width-1]:
                            for y in [0,height-1]:
                                if im_mat[y,x] > 237:
                                    white_corners+=1
                        if white_corners == 4:
                            isgood = False
                        if width < 256:
                            isgood = False
                        if height < 256:
                            isgood = False
                    except:
                        isgood = False
                    if im in ims_used:
                        isgood = False
                    if isgood:
                        wnid = im.split('_')[0]
                        n_read += 1
                        if wnid in to_sample[wnid2cat(wnid,wnidMap)].keys():
                            to_sample[wnid2cat(wnid,wnidMap)][wnid].append(im_fn)
                        else:
                            to_sample[wnid2cat(wnid,wnidMap)][wnid] = []
                    if n_read%1000 == 0:
                            print str(n_read)+' images read'
            #Non-animal distractors
            for fold in os.listdir(os.path.join(base_extra,'nonanimal_extra')):
                for im in os.listdir(os.path.join(base_extra,'nonanimal_extra',fold)):
                    im_fn = os.path.join(base_extra,'nonanimal_extra',fold,im)            #verfiy
                    isgood = True
                    #see if file exists
                    if not(os.path.isfile(im_fn)):
                        isgood = False
                    #see if it's an openable image file
                    try:
                        img = Image.open(im_fn)
                        width, height = img.size
                        white_corners = 0
                        im_mat = np.asarray(img.convert('L'))
                        for x in [0,width-1]:
                            for y in [0,height-1]:
                                if im_mat[y,x] > 237:
                                    white_corners+=1
                        if white_corners == 4:
                            isgood = False
                        if width < 256:
                            isgood = False
                        if height < 256:
                            isgood = False
                    except:
                        isgood = False
                    if im in ims_used:
                        isgood = False
                    if isgood:
                        wnid = im.split('_')[0]
                        n_read += 1
                        if wnid in to_sample[wnid2cat(wnid,wnidMap)].keys():
                            to_sample[wnid2cat(wnid,wnidMap)][wnid].append(im_fn)
                        else:
                            to_sample[wnid2cat(wnid,wnidMap)][wnid] = []
                    if n_read%1000 == 0:
                            print str(n_read)+' images read'
            pickle.dump(to_sample,open('extra_ims.p','wb'))
        else:
            to_sample = pickle.load(open('extra_ims.p','rb'))
        #move appropriate number of images
        ims_used = test_ims
        for cat in to_write.keys():
            for im in to_write[cat]:
                ims_used.append(im)
        
        for cat in to_sample.keys():
            print 'sampling '+cat
            if cat == 'mammals':
                for wnid in to_sample[cat].keys():
                    for im in to_sample[cat][wnid]:
                        im_id = im.split('/')[-1]
                        if im_id not in ims_used and os.path.exists(im):
                            shutil.move(im,os.path.join(outdir,im_id))            
            else:
                wnids = to_sample[cat].keys()
                to_find = (tot_num/4)-len(to_write[cat])
                found = len(to_write[cat])
                while to_find > 0:
                    seed = random.randint(0,len(wnids)-1)
                    curr_wnid = wnids[seed]
                    if len(to_sample[cat][curr_wnid]) > 0:
                        if len(to_sample[cat][curr_wnid]) ==1:
                            im_seed = 0
                        else:
                            im_seed = random.randint(0,len(to_sample[cat][curr_wnid])-1)
                        im = to_sample[cat][curr_wnid][im_seed]
                        im_id = im.split('/')[-1]
                        if im_id not in ims_used and os.path.exists(im):
                            ims_used.append(im_id)
                            found +=1
                            to_find -= 1
                            shutil.move(im,os.path.join(outdir,im_id))
                            to_write[cat].append(im_id)
                            if found%1000 == 0:
                                    print str(found)+'/'+str(tot_num/4)+' '+cat+' images found'
        pickle.dump(to_write,open('complete_list.p','wb'))
    else:
        if not os.path.exists('ims_for_train.p'):
            print 'reading in images'
            in_fold = {}
            in_fold['mammals'] = []
            in_fold['nonmammals'] = []
            in_fold['natural'] = []
            in_fold['artificial'] = []
            count = 0
            for im in os.listdir(outdir_train):
                im_path = os.path.join(outdir_train,im)
                if os.path.exists(im_path):
                    wnid = im.split('_')[0]
                    in_fold[wnid2cat(wnid,wnidMap)].append(im)
                    count += 1
                    if count%1000 == 0:
                        print str(count)+' images read'
            pickle.dump(in_fold,open('ims_for_train.p','wb'))
        else:
            in_fold = pickle.load(open('ims_for_train.p','rb'))
            np.random.shuffle(in_fold['mammals'])
            np.random.shuffle(in_fold['nonmammals'])
            np.random.shuffle(in_fold['natural'])
            np.random.shuffle(in_fold['artificial'])
            n_sets = tot_num/5000
            for i in range(n_sets):
                temp_set = []
                for cat in in_fold.keys():
                    for im_num in range(i*(5000/4),(i+1)*(5000/4)):
                        temp_set.append(in_fold[cat][im_num])
            
                f = open(os.path.join(base_extra,'sets','trainset_'+str(i)+'.txt'),'w')
                np.random.shuffle(temp_set)
                for im in temp_set:
                    wnid = im.split('_')[0]
                    cat = wnid2cat(wnid,wnidMap)
                    if (cat == 'mammals') or (cat == 'nonmammals'):
                        lab = 1
                    else:
                        lab = 0
                    f.write(im+'\t'+str(lab)+'\n')
                f.close()
def prep_train(test_fp, base_extra,tot_num):
    #crop images
    im_list = []
    n_sets = tot_num/5000
    turk_out = os.path.join('/media/data_cifs/nsf_levels/TURK_IMS/trainset')
    if not os.path.exists(turk_out):
        os.mkdir(turk_out)
    for i in range(n_sets):
        f = open(os.path.join(base_extra,'sets','trainset_'+str(i)+'.txt'),'r')
        for line in f:
            im_id = line.split()[0]
            im_path = os.path.join(base_extra,'sets','trainingset_v',im_id)
            im_list.append(im_path)
        f.close()
    for im_path in im_list: 
        if os.path.exists(im_path):
            img = Image.open(im_path)
            im_id = im_path.split('/')[-1]
            width, height = img.size
            # center crop to square
            if not((width == 256) and (height == 256)):
                if width > height:
                    left = (width - height)/2
                    top = 0
                    right = left + height
                    bottom = top + height
                else:
                    left = 0 
                    top = (height - width)/2
                    right = left + width
                    bottom = top + width 
                img = img.crop((left, top, right, bottom))
                # resize
                img = img.resize((256,256),Image.ANTIALIAS)
            # save
            img.convert('RGB').save(os.path.join(turk_out,im_id)+'.jpg')
            
def amt_train(test_fp, base_extra,im_fp, tot_num):
    #crop images
    print 'Converting to grayscale'
    n_sets = tot_num/5000
    turk_out = os.path.join('/media/data_cifs/nsf_levels/TURK_IMS/trainset_bw')
    if not os.path.exists(turk_out):
        os.mkdir(turk_out)
    count = 0
    for im in os.listdir(im_fp):
        im_path = os.path.join(im_fp,im)
        img = Image.open(im_path)
        width, height = img.size
        #convert to Grayscale
        im_mat = np.asarray(img.convert('L'))
        # save
        outdir = os.path.join(turk_out,im)
        im_tosave = Image.fromarray(im_mat).convert('L')
        im_tosave.save(os.path.join(outdir))      
        count += 1
        if count%1000 == 0:
            print str(count)+' images read'
            
        
                    
def update_sampled_list(im_txt):
    sampled_list = []
    
    train_ims = pickle.load(open('/home/jcader/Thesis/emtcat/ims_for_train.p','rb'))
    for key in train_ims.keys():
        for im in train_ims[key]:
            sampled_list.append(im)
    f = open(im_txt,'r')
    for line in f:
        im_id = line.split()[0]
        sampled_list.append(im_id)
    f.close()
    #Ex ims
    sampled_list.append('n01448951_2526')
    sampled_list.append('n01616086_3881')
    sampled_list.append('n02080713_1804')
    sampled_list.append('n02366002_1068')
    sampled_list.append('n02793199_3013')
    sampled_list.append('n03318294_4349')       
    sampled_list.append('n12451399_7975')
    sampled_list.append('n13107891_13009')
    
    pickle.dump(sampled_list,open('/home/jcader/Thesis/emtcat/sampled_list1603241729','wb'))
        
        

        
    
          
            
im_dir = '/media/data/nsf_levels'
labels_path = '/home/jcader/Thesis/emtcat/fall11_urls.txt'
root = get_synset_root('/home/jcader/Thesis/emtcat/ImageNet_structure.xml')
wnidMap = make_wnimdMap(root)
#setid = '1603241729'
setid='1603241729'
im_txt = '/media/data/nsf_levels/raw_ims/sets/set1603241729_0.txt'
outdir_train = '/media/data_cifs/nsf_levels/raw_ims/sets/trainingset_v'
base_down = '/media/data_cifs/nsf_levels/raw_ims/sets/trainingset'
base_extra = '/media/data_cifs/nsf_levels/raw_ims'

#download_rep_ims(root,im_dir,labels_path,False, False,setid)
#sample_set(1,300,setid,30,im_dir,wnidMap)
verify(setid,30,300,im_dir,wnidMap,True)
process_for_AMT(setid,30,im_dir)
#save_norm_for_AMT(setid,0,im_dir)

#make_new_train(wnidMap,setid,base_extra,base_down,im_txt,outdir_train,100000,False) 
#prep_train('tktkt', base_extra,100000)
#amt_train('tktkt', base_extra,'/media/data_cifs/nsf_levels/TURK_IMS/trainset',100000)

