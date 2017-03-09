import sqlite3
import json
import numpy as np
import pylab as P
import pickle
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from results_key import label_results
import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import decimal

r=sqlite3.connect('/home/jcader/Thesis/emtcat/emtcat_participants_0508.db').cursor().execute("SELECT workerid,beginhit,status,datastring FROM placecat WHERE status in (3,4) AND NOT datastring==''").fetchall()
print "%d participants found." % len(r)

# Generate ground truth key
im2lab = label_results('/media/data/nsf_levels/raw_ims/sets/set1603241729_0.txt')
data_by_subj_by_rt = {}
trials_by_img = {} # trial data indexed by image
data_by_img = {} # result data averaged over trials by image
model_comp = {} # data for model comparison
model_comp['animal'] = {}
model_comp['nonanimal'] = {}
im2key = {} # maps image to index in model comp array
animal_ind = 0;
nonanimal_ind =0;
# store image names with index in dict to hold model comparisons
for im in im2lab.keys():
    if im2lab[im] == 'animal':
        im2key[im] = animal_ind
        animal_ind +=1
    else:
        im2key[im] = nonanimal_ind
        nonanimal_ind +=1
model_comp['animal']['source_images'] = np.empty([animal_ind,1],dtype='S20')
model_comp['nonanimal']['source_images'] = np.empty([nonanimal_ind,1],dtype='S20')
for im in im2key.keys():
    model_comp[im2lab[im]]['source_images'][im2key[im]][0] = im
model_comp['animal']['human_acc'] = np.empty([1,animal_ind]) #arrays to hold human accuracy data for model comparison      
model_comp['nonanimal']['human_acc'] = np.empty([1,nonanimal_ind])


data_by_max_rt = {} #results by max response time
data_by_max_rt['500'] =[[]for i in range(50)]
data_by_max_rt['1000'] =[[]for i in range(50)]
data_by_max_rt['2000'] =[[]for i in range(50)]
responses_by_max_rt = {} #results by max response time
responses_by_max_rt['500'] =[[]for i in range(50)]
responses_by_max_rt['1000'] =[[]for i in range(50)]
responses_by_max_rt['2000'] =[[]for i in range(50)]
responses_by_max_rt['2000'] =[[]for i in range(50)]
acc_by_im = {}
avg_acc_by_im = {}
total_acc = []
total_time = []
min_rt_subj = {}
min_rt_subj['animal'] = {}
min_rt_subj['nonanimal'] = {}
min_rt_subj['animal_wrong'] = {}
min_rt_subj['nonanimal_wrong'] = {}
im_by_subj= {}

myfile = open('mtresults.txt','w')
trial_ct = 0
for i_subj in range(0,len(r)):

    data = json.loads(r[i_subj][3])
    im_by_subj[i_subj] = {}
    # Get experimental trials.
    alltrials=data['data']
    # Ignore training trials
    exptrials=alltrials[10:]
    # Ignore pause screens
    trials = [t['trialdata'] for t in exptrials if t['trialdata']['trial_type'] == 's2stim']
    print "subj %d(%d trials) = %s" % (i_subj, len(trials), r[i_subj][0])
    
    # Store data by subj 
    min_rt_subj['animal'][i_subj] = []
    min_rt_subj['nonanimal'][i_subj] = []
    min_rt_subj['animal_wrong'][i_subj] = []
    min_rt_subj['nonanimal_wrong'][i_subj] = []
    # Organize trials per-image
    count = 0
    for t in trials:
        stim = json.loads(t["stimulus"])
        #recover max_rt and im num in block
        max_rt = stim["duration"]-stim["onset"]-50
        im_in_block = count%(len(trials)/6)
        t['max_rt'] = max_rt
        t['im_in_block'] = im_in_block
        img = stim["stimulus"].split('/')[3]
        t['cat'] = stim["stimulus"].split('/')[3]
        # Get score 1/0 for true/wrong answer for accumulating accuracies
        if t['response'] == t['true_response']:
            t['score'] = 1
            if t['rt'] > 0:
                min_rt_subj[im2lab[(t['cat']+'.jpg')]][i_subj].append(t['rt'])
        else:
            t['score'] = 0
            if t['rt'] > 0:
                min_rt_subj[im2lab[(t['cat']+'.jpg')]+'_wrong'][i_subj].append(t['rt'])
            
        im_by_subj[i_subj][img] =  t['score']   
        if t['rt'] < 0:
            t['cscore'] = 0.5
        else:
            t['cscore'] = t['score']
        if img in trials_by_img:
            trials_by_img[img] += [t]
        else:
            trials_by_img[img] = [t]
        trial_ct +=1
        if t['rt'] > 0:
            total_time.append(t['rt'])
        total_acc.append(t['cscore']) #used to be indented #score or tscore
        #store data by max_rt
        data_by_max_rt[str(max_rt)][im_in_block].append(t['rt'])
        responses_by_max_rt[str(max_rt)][im_in_block].append(t['cscore']) #score or tscore
        #Store per image data
        if img not in acc_by_im.keys():
            acc_by_im[img] = {}
            avg_acc_by_im[img] = [None,None,None]
        if str(max_rt) not in acc_by_im[img].keys():
            acc_by_im[img][str(max_rt)] = []
        acc_by_im[img][str(max_rt)].append(t['score'])
        count += 1
        #store for double bootstrapping
        if i_subj not in data_by_subj_by_rt.keys():
            data_by_subj_by_rt[i_subj] = {}
            data_by_subj_by_rt[i_subj]['500'] = {}
            data_by_subj_by_rt[i_subj]['1000-2000'] = {}
        if max_rt < 1000:
            key = '500'
        else:
            key = '1000-2000'
        if img not in data_by_subj_by_rt[i_subj][key].keys():
            data_by_subj_by_rt[i_subj][key][img] = []
        data_by_subj_by_rt[i_subj][key][img].append(t['cscore'])
            
	
# caluclate mean RT and performance per image
for img,trials in trials_by_img.iteritems():
    n = len(trials)
    score = float(sum([t['score'] for t in trials])) / n
    cscore = float(sum([t['cscore'] for t in trials])) / n
    rt = 0.0
    n_valid = 0
    for t in trials:
        if t['rt'] >= 0:
            rt += t['rt']
            n_valid += 1
    if n_valid == 0:
        n_valid = 1
    rt /= n_valid
    data = dict()
    data['score'] = score
    data['cscore'] = cscore
    data['n'] = n
    data['cat'] = trials[0]['cat']
    data['rt'] = rt
    data_by_img[img] = data
    #print "Image %s has %d trials and %.1f%% correct." % (img, n, score * 100)
    myfile.write(img+' '+str(score*100)+' '+str(rt)+'\n')
    # Store data for model comparison
    if im2lab[img+'.jpg'] == 'animal':
        model_comp['animal']['human_acc'][0][im2key[img+'.jpg']] = float(cscore*100)
    else:
        model_comp['nonanimal']['human_acc'][0][im2key[img+'.jpg']] = float(cscore*100)

myfile.close()
score_data = [data_by_img[img]['score']*100 for img in data_by_img]
rt_data = [data_by_img[img]['rt'] for img in data_by_img]

pickle.dump(trials_by_img, open('c_trials_by_img_b.p', 'wb'))
pickle.dump(data_by_img, open('c_data_by_img_b.p', 'wb'))
#sio.savemat('trials_by_img_c.mat', trials_by_img)
#sio.savemat('c_data_by_img_b.mat', data_by_img)

#Analyze and graph results
# Track failure rates 
failures = [[0,0], [0,0], [0,0]]
# rt hists by max_rt
mrt_500 = []
mrt_1000 = []
mrt_2000 = []
mrt_500_25 = []
mrt_1000_25 = []
mrt_2000_25 = []
mrt_500_class = []
mrt_1000_class = []
mrt_2000_class = []
mrt_500_25_class = []
mrt_1000_25_class = []
mrt_2000_25_class = []
for im_num in range(len(data_by_max_rt['500'])):
    for trial in range(len(data_by_max_rt['500'][0])):
        failures[0][1] +=1
        failures[1][1] +=1
        failures [2][1] +=1
        if data_by_max_rt['500'][im_num][trial] > 0:
            mrt_500.append(data_by_max_rt['500'][im_num][trial])
            if im_num >23:
                mrt_500_25.append(data_by_max_rt['500'][im_num][trial])
        else:
            failures[0][0] +=1
        mrt_500_class.append(responses_by_max_rt['500'][im_num][trial])
        if im_num >23:
                mrt_500_25_class.append(responses_by_max_rt['500'][im_num][trial])

        if data_by_max_rt['1000'][im_num][trial] > 0:
            mrt_1000.append(data_by_max_rt['1000'][im_num][trial])
            if im_num >23:
                mrt_1000_25.append(data_by_max_rt['1000'][im_num][trial])
        else:
            failures[1][0] +=1
        mrt_1000_class.append(responses_by_max_rt['1000'][im_num][trial])
        if im_num >23:
            mrt_1000_25_class.append(responses_by_max_rt['1000'][im_num][trial])
            
        if data_by_max_rt['2000'][im_num][trial] > 0:
            mrt_2000.append(data_by_max_rt['2000'][im_num][trial])
            if im_num >23:
                mrt_2000_25.append(data_by_max_rt['2000'][im_num][trial])
        else:
            failures[2][0] +=1
        mrt_2000_class.append(responses_by_max_rt['2000'][im_num][trial])
        if im_num >23:
            mrt_2000_25_class.append(responses_by_max_rt['2000'][im_num][trial])

mrt_500_avgrt = []
mrt_1000_avgrt = []
mrt_2000_avgrt = []
mrt_500_avgclass = []
mrt_1000_avgclass = []
mrt_2000_avgclass = []													
# Average rt over time
for im_num in range(len(responses_by_max_rt['500'])):
    temp_500 = []
    temp_1000 = []
    temp_2000 = []
    temp_500_class = []
    temp_1000_class = []
    temp_2000_class = []
    for trial in range(len(responses_by_max_rt['500'][0])):
        if data_by_max_rt['500'][im_num][trial] > 0:
            temp_500.append(data_by_max_rt['500'][im_num][trial])
            temp_500_class.append(responses_by_max_rt['500'][im_num][trial])
        if data_by_max_rt['1000'][im_num][trial] > 0:
            temp_1000.append(data_by_max_rt['1000'][im_num][trial])
            temp_1000_class.append(responses_by_max_rt['1000'][im_num][trial])
        if data_by_max_rt['2000'][im_num][trial] > 0:
            temp_2000.append(data_by_max_rt['2000'][im_num][trial])
            temp_2000_class.append(responses_by_max_rt['2000'][im_num][trial])
    mrt_500_avgrt.append(np.mean(temp_500))
    mrt_1000_avgrt.append(np.mean(temp_1000))
    mrt_2000_avgrt.append(np.mean(temp_2000))
    mrt_500_avgclass.append(np.mean(temp_500_class)*100)
    mrt_1000_avgclass.append(np.mean(temp_1000_class)*100)
    mrt_2000_avgclass.append(np.mean(temp_2000_class)*100)
# Average accuracy per image by max_rt
for im in acc_by_im.keys():
    avg_acc_by_im[im][0] = np.mean(acc_by_im[im]['500'])*100
    avg_acc_by_im[im][1] = np.mean(acc_by_im[im]['1000'])*100
    avg_acc_by_im[im][2] = np.mean(acc_by_im[im]['2000'])*100
									
# Find failure rates
fail_500 = float(failures[0][0])*100/float(failures[0][1])
fail_1000 = float(failures[1][0])*100/float(failures[1][1])
fail_2000 = float(failures[2][0])*100/float(failures[2][1])

#---Failure Rates---
#f = P.figure()
#y = [fail_500,fail_1000,fail_2000]
#x = range(3)
#ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
#bg = ax.bar(x, y, align='center')
#ax.set_xticks(x)
#ax.set_xticklabels(['500ms', '1000ms', '2000ms'])
#P.title('Failure Rates by Max Response Time')
#ax.set_ylim([0,100])
#ax.set_ylabel('Percent of Trials Timed-Out')
#ax.set_xlabel('Max Response Time')
##label
#for mrt in bg:
#    height = float(mrt.get_height())
#    print height
#    ax.text(mrt.get_x() + mrt.get_width()/2., 1.05*height,str(height)+'%', ha='center', va='bottom')
#										
#
##---rt distribution by class---
#binwidth = 50
#f, ((pl500, pl1000, pl2000),(pl501, pl1001, pl1501))= P.subplots(2, 3, sharey='row')
#
#P.figure()
#pl500.hist(mrt_500,bins=np.arange(min(mrt_500), max(mrt_500) + binwidth, binwidth))
#pl500.set_xlabel('Reaction Times (ms)')
#pl500.set_ylabel('# Images')
#pl500.axis([0,1600,0,500])
#pl500.set_title('500ms Max Response Time')
#P.savefig('c_turk_hist50025rt.png')
#print 'Average 500: '+str(np.mean(mrt_500))+' , Stdev: '+str(np.std(mrt_500)) 
#
#P.figure()
#pl1000.hist(mrt_1000,bins=np.arange(min(mrt_1000), max(mrt_1000) + binwidth, binwidth))
#pl1000.set_xlabel('Reaction Times (ms)')
#pl1000.axis([0,1600,0,500])
#pl1000.set_ylabel('# Images')
#pl1000.set_title('1000ms Max Response Time')
#P.savefig('c_turk_hist100025rt.png')
#print 'Average 1000: '+str(np.mean(mrt_1000))+' , Stdev: '+str(np.std(mrt_1000))
#
#P.figure()
#pl2000.hist(mrt_2000,bins=np.arange(min(mrt_2000), max(mrt_2000) + binwidth, binwidth))
#pl2000.set_xlabel('Reaction Times (ms)')
#pl2000.set_ylabel('# Images')
#pl2000.axis([0,1600,0,500])
#pl2000.set_title('2000ms Max Response Time')
#P.savefig('c_turk_hist200025rt.png')
#print 'Average 2000: '+str(np.mean(mrt_2000))+' , Stdev: '+str(np.std(mrt_2000))
#f.suptitle('Distribution of Classifcation Rates for Varied Max Response Times')
##P.savefig('c_turk_histsrt')

#--- distribution by class---

f = P.figure()
x = xrange(3)
y = [np.mean(mrt_500_class)*100, np.mean(mrt_1000_class)*100, np.mean(mrt_2000_class)*100]
stdevs = [np.std(mrt_500_class), np.std(mrt_1000_class), np.std(mrt_2000_class)]
ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
bg = ax.bar(x, y, align='center', yerr=stdevs, ecolor ='k')
ax.set_xticks(x)
ax.set_xticklabels(['500ms', '1000ms', '2000ms'])
P.title('Accuracy by Max Response Time')
ax.set_ylim([0,100])
ax.set_ylabel('Categorization Accuracy (%)')
ax.set_xlabel('Max Response Time')
#label
for mrt in bg:
    height = mrt.get_height()
    ax.text(mrt.get_x() + mrt.get_width()/2., 1.05*height,'%d' % int(height)+'%', ha='center', va='bottom')

#P.savefig('c_turk_class.png')

#--- distribution last 25 by class---

#f = P.figure()
#x = xrange(3)
#y = [np.mean(mrt_500_25_class)*100, np.mean(mrt_1000_25_class)*100, np.mean(mrt_2000_25_class)*100]
#stdevs = [np.std(mrt_500_25_class), np.std(mrt_1000_25_class), np.std(mrt_2000_25_class)]
#ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
#bg = ax.bar(x, y, align='center', yerr=stdevs, ecolor ='k')
#ax.set_xticks(x)
#ax.set_xticklabels(['500ms', '1000ms', '2000ms'])
#P.title('Accuracy by Max Response Time')
#ax.set_ylim([0,100])
#ax.set_ylabel('Categorization Accuracy (%)')
##label
#for mrt in bg:
#    height = mrt.get_height()
#    ax.text(mrt.get_x() + mrt.get_width()/2., 1.05*height,'%d' % int(height)+'%', ha='center', va='bottom')
#P.savefig('c_turk_class_25.png')
#    
#---rt distribution by maxrt, last 25--

#binwidth = 50
#f, (pl500, pl1000, pl2000) = P.subplots(1, 3, sharey=True)
#P.figure()
#pl501.hist(mrt_500_25,bins=np.arange(min(mrt_500_25), max(mrt_500_25) + binwidth, binwidth))
#pl501.set_xlabel('Reaction Times (ms)')
#pl501.set_ylabel('# Images')
#pl501.axis([0,1600,0,300])
#pl501.set_title('500ms Max Response Time')
#P.savefig('c_turk_hist50025rt.png')
#print 'Average 500: '+str(np.mean(mrt_500_25))+' , Stdev: '+str(np.std(mrt_500_25))

#P.figure()
#pl1001.hist(mrt_1000_25,bins=np.arange(min(mrt_1000_25), max(mrt_1000_25) + binwidth, binwidth))
#pl1001.set_xlabel('Reaction Times (ms)')
#pl1001.axis([0,1600,0,300])
#pl1000.set_ylabel('# Images')
#pl1001.set_title('1000ms Max Response Time')
#P.savefig('c_turk_hist100025rt.png')
#print 'Average 1000: '+str(np.mean(mrt_1000_25))+' , Stdev: '+str(np.std(mrt_1000_25))

#P.figure()
#pl1501.hist(mrt_2000_25,bins=np.arange(min(mrt_2000_25), max(mrt_2000_25) + binwidth, binwidth))
#pl1501.set_xlabel('Reaction Times (ms)')
#pl2000.set_ylabel('# Images')
#pl1501.axis([0,1600,0,300])
#pl1501.set_title('2000ms Max Response Time')
#P.savefig('c_turk_hist200025rt.png')
#print 'Average 2000: '+str(np.mean(mrt_2000_25))+' , Stdev: '+str(np.std(mrt_2000_25))
#f.suptitle('Distribution of Classifcation Rates for Varied Max Response Times (Last 25 in Block)')
#P.savefig('c_turk_hists_25rt')
#--- Summary rt and class ---
#P.figure()
#P.hist(score_data)
#P.xlabel('Percent Correctly Classified (%)')
#P.ylabel('# Images')
#P.title('Distribution of Classification Rate by Image .')
#P.savefig('c_turk_histclass.png')


#P.figure()
#P.hist(rt_data)
#P.xlabel('Reaction time [ms]')
#P.ylabel('# images')
#P.title('Distribution of reaction times by image by AMT users.')
#P.savefig('c_turk_histrt.png')
#print "Average accuracy:", np.mean(total_acc), "Average rt:", np.mean(total_time)
#print (1-((float(len(total_time)))/float(trial_ct)))*100, "% discarded"
#--- Average rt over time ---
#
#P.figure()
##find trends
#fit = np.polyfit(range(1,51),mrt_500_avgrt,1)
#trend = np.poly1d(fit) 
#fit = np.polyfit(range(1,51),mrt_1000_avgrt,1)
#trend1 = np.poly1d(fit)
#fit = np.polyfit(range(1,51),mrt_2000_avgrt,1)
#trend2 = np.poly1d(fit)
#P.scatter(range(50),mrt_500_avgrt, c='b',label='500ms Max RT')
#P.hold = True
#P.plot(range(1,50),trend(range(1,50)), c='b')
#P.hold = True
#P.scatter(range(50),mrt_1000_avgrt, c='r',label='1000ms Max RT')
#P.hold = True
#P.plot(range(1,50),trend1(range(1,50)), c='r')
#P.hold = True
#P.scatter(range(50),mrt_2000_avgrt, c='g',label='2000ms Max RT')
#P.plot(range(1,50),trend2(range(1,50)), c='g')
#P.ylabel('Reaction Time (ms)')
#P.xlabel('Image # in Block')
#P.title('Average Reaction Times Over 50 Image Test Block')
#P.legend(bbox_to_anchor=(1, .5))
#P.xlim([-.5,51])
#P.ylim([400,600])
##--- Average accuracy over time ---
#
#P.figure()
##find trends
#fit = np.polyfit(range(1,51),mrt_500_avgclass,1)
#trend = np.poly1d(fit) 
#fit = np.polyfit(range(1,51),mrt_1000_avgclass,1)
#trend1 = np.poly1d(fit)
#fit = np.polyfit(range(1,51),mrt_2000_avgclass,1)
#trend2 = np.poly1d(fit)
#P.scatter(range(50),mrt_500_avgclass, c='b',label='500ms Max RT')
#P.hold = True
#P.plot(range(1,50),trend(range(1,50)), 'b')
#P.hold = True
#P.scatter(range(50),mrt_1000_avgclass, c='r',label='1000ms Max RT')
#P.hold = True
#P.plot(range(1,50),trend1(range(1,50)), 'r')
#P.hold = True
#P.scatter(range(50),mrt_2000_avgclass, c='g',label='2000ms Max RT')
#P.plot(range(1,50),trend2(range(1,50)), 'g')
#P.ylabel('Classification Accuracy (%)')
#P.xlabel('Image # in Block')
#P.title('Average Accuracy Over 50 Image Test Block')
#P.legend(bbox_to_anchor=(1, 0.2))
#P.xlim([-.5,51])

#P.savefig('c_turk_classtime.png')

#--- Look at Variance ---
# for classification
#f_val, p_val = stats.f_oneway(mrt_500, mrt_1000, mrt_2000) 
#print "RT: One-way ANOVA P =", p_val, "F =", f_val
#f_val, p_val = stats.f_oneway(mrt_500_class, mrt_1000_class, mrt_2000_class) 
#print "RT: One-way ANOVA P =", p_val, "F =", f_val
#post-hoc correction 
#for reaction times
#compile into one data set
#dset = []
#ind = 1
#for i in range(len(mrt_500)):
#    dset.append((ind, '500', mrt_500[i]))
#    ind += 1
#for i in range(len(mrt_1000)):
#    dset.append((ind,'1000',  mrt_1000[i]))
#    ind += 1
#for i in range(len(mrt_2000)):
#    dset.append((ind, '2000', mrt_2000[i]))
#    ind += 1
#dset = np.rec.array(dset,dtype=[('ind', '<i5'),('mrt_group','|S5'),('mrt','<i5')])
#thsd = pairwise_tukeyhsd(dset['mrt'],dset['mrt_group'])
#print thsd
#
###for RT
#f_val, p_val = stats.f_oneway(mrt_500, mrt_1000, mrt_2000) 
#print "RT: One-way ANOVA P =", p_val, "F =", f_val
#post-hoc correction 
#for reaction times
#compile into one data set
#dset = []
#ind = 1
#for i in range(len(mrt_500)):
#    dset.append((ind, '500', mrt_500[i]))
#    ind += 1
#for i in range(len(mrt_1000)):
#    dset.append((ind,'1000',  mrt_1000[i]))
#    ind += 1
#for i in range(len(mrt_2000)):
#    dset.append((ind, '2000', mrt_2000[i]))
#    ind += 1
#dset = np.rec.array(dset,dtype=[('ind', '<i5'),('mrt_group','|S5'),('mrt','<i5')])
#thsd = pairwise_tukeyhsd(dset['mrt'],dset['mrt_group'])
#print thsd
#
#
#for accuracy 
#dset2 = []
#ind = 1
#for i in range(len(mrt_500_class)):
#    dset2.append((ind, '500', mrt_500_class[i]))
#    ind += 1
#for i in range(len(mrt_1000_class)):
#    dset2.append((ind,'1000',  mrt_1000_class[i]))
#    ind += 1
#for i in range(len(mrt_2000_class)):
#    dset2.append((ind, '2000', mrt_2000_class[i]))
#    ind += 1
#dset2 = np.rec.array(dset2,dtype=[('ind', '<i5'),('mrt_group','|S5'),('accuracy','<i5')])
#thsd = MultiComparison(dset2['accuracy'],dset2['mrt_group'])
#print thsd.tukeyhsd()
##look up group indicies 
#print thsd.groupsunique

#--- Split into easy, medium, hard ---
#P.figure()
#count = 0
#e_c = 0
#m_c = 0
#h_c = 0
#n_c = 0
#key = []
#easy = []
#medium = []
#hard = []
#unreliable = []
#for im in avg_acc_by_im.keys():
    #if (avg_acc_by_im[im][0] > 75) and (avg_acc_by_im[im][1] > 75) and (avg_acc_by_im[im][2] > 75):
        #P.scatter([500,1000,2000],avg_acc_by_im[im], color='g')
        #e_c+=1
        #easy.append([np.mean(avg_acc_by_im[im]), count])
        #key.append(im)
    #elif (avg_acc_by_im[im][1] > 75) and (avg_acc_by_im[im][2] > 75) :  
        #P.scatter([500,1000,2000],avg_acc_by_im[im], color='y')
        #m_c +=1
        #medium.append([np.mean(avg_acc_by_im[im]),count])
        #key.append(im)
    #elif (avg_acc_by_im[im][2] > 75):
        #P.scatter([500,1000,2000],avg_acc_by_im[im], color='r')
        #h_c+=1
        #hard.append([np.mean(avg_acc_by_im[im]), count])
        #key.append(im)
    #else:
        #P.scatter([500,1000,2000],avg_acc_by_im[im], color='b')
        #unreliable.append([count,np.mean(avg_acc_by_im[im]), count])
        #key.append(im)
        #n_c +=1
    #count +=1
    #P.hold = True
#P.scatter([500,1000,2000],avg_acc_by_im[im], color='g',label='Easy')
#P.scatter([500,1000,2000],avg_acc_by_im[im], color='y',label='Medium')
#P.scatter([500,1000,2000],avg_acc_by_im[im], color='r',label='Hard')
#P.scatter([500,1000,2000],avg_acc_by_im[im], color='b',label='N/A')
#P.legend(bbox_to_anchor=(.8, 0.3))
#P.ylabel('Classification Accuracy (%)')
#P.xlabel('Max Response Time (ms)')
#P.title('Reliable Categorization ( >75%) at Varied Max Response Times')
#print "Easy images:",e_c, "Medium Images:", m_c, "Hard images:", h_c, "Unreliable:", n_c
#easy.sort(reverse=True)
#medium.sort(reverse=True)
#hard.sort(reverse=True)

#--- find min rt per subject ---
# for animal
bin_width = 50
maxrt = 2000
minrt = 0
y_min = -20
y_max = 45
min_rt_hists = {}
min_rt_hists['animal'] = {}
min_rt_hists['nonanimal'] = {}
min_rt_hists['animal_wrong'] = {}
min_rt_hists['nonanimal_wrong'] = {}
min_rt_ratio = {}
min_rt_ratio['animal'] = {}
min_rt_ratio['nonanimal'] = {}
# for animals
for i in range(len(min_rt_subj['animal'].keys())):  
    #plt.subplot(4,5,i)
    #plt.xlim(minrt,maxrt)
    min_rt_hists['animal_wrong'][i] = np.histogram(min_rt_subj['animal_wrong'][i],bins=maxrt/bin_width,range=(minrt,maxrt))[0]
    min_rt_hists['animal'][i] = np.histogram(min_rt_subj['animal'][i],bins =30,range=(minrt,maxrt))[0]
    #min_rt_hists['animal'][i] = plt.hist(min_rt_subj['animal'][i],bins =30,range=(minrt,maxrt))[0]
    #min_rt_hists['animal_wrong'][i] = plt.hist(min_rt_subj['animal_wrong'][i],bins=maxrt/bin_width,range=(minrt,maxrt))[0]
    #plt.title('subject '+str(i))
    #plt.xlabel('Reaction Time (ms)')
#plt.suptitle('Correct Identification of Animal Images')
#plt.close()
# for nonanimals
#plt.figure()    
for i in range(len(min_rt_subj['nonanimal'].keys())):  
    #plt.subplot(4,5,i)
    #plt.xlim(minrt,maxrt)
#min_rt_hists['nonanimal'][i] = plt.hist(min_rt_subj['nonanimal'][i],bins =30,range=(minrt,maxrt))[0]
    min_rt_hists['nonanimal_wrong'][i] = np.histogram(min_rt_subj['nonanimal_wrong'][i],bins=maxrt/bin_width,range=(minrt,maxrt))[0]
    min_rt_hists['nonanimal'][i] = np.histogram(min_rt_subj['nonanimal'][i],bins =30,range=(minrt,maxrt))[0]
    #min_rt_hists['nonanimal_wrong'][i] = plt.hist(min_rt_subj['nonanimal_wrong'][i],bins=maxrt/bin_width,range=(minrt,maxrt))[0]
    #plt.title('subject '+str(i))
    #plt.xlabel('Reaction Time (ms)')
#plt.suptitle('Correct Identification of Non-Animal Images')
#plt.close()
min_rts = {}
min_rts['animal'] = {}
min_rts['nonanimal'] = {}
min_rts['animal_p'] = {}
min_rts['nonanimal_p'] = {}
# Plot ratios
#fig = plt.figure()
#for i in range(len(min_rt_subj['nonanimal'].keys())):
#    #for animal
#    #graph successes
#    ax = fig.add_subplot(4,5,i+1)
#    plt.xlim(minrt,maxrt)
#    plt.ylim(y_min,y_max)
#    ylabs,ylocs = plt.yticks()
#    plt.yticks(np.arange(y_min,y_max,len(ylocs)),abs(ylabs).astype(int))
#    plt.title('Subject '+str(i))
#    plt.xlabel('RT (ms)')
#    plt.ylabel('Number of Images')
#    # find min rt
#    for bin in range(len(min_rt_hists['animal'][i])):
#        if stats.binom_test([min_rt_hists['animal'][i][bin],min_rt_hists['animal_wrong'][i][bin]]) < .05:
#            min_rts['animal'][i] = (bin*bin_width+(bin_width/2),min_rt_hists['animal'][i][bin])
#            min_rts['animal_p'][i] = stats.binom_test([min_rt_hists['animal'][i][bin],min_rt_hists['animal_wrong'][i][bin]])
#            break
#    plt.bar(np.arange(minrt,maxrt,bin_width),min_rt_hists['animal'][i], width=bin_width,hold=True, color='g',label='Correct')
#    plt.bar(np.arange(minrt,maxrt,bin_width),min_rt_hists['animal_wrong'][i]*-1, width=bin_width,hold=True, color='r', label='Incorrect')
#    plt.plot((min_rts['animal'][i][0],min_rts['animal'][i][0]),(y_min,y_max),'b--',label='Min RT')
#    plt.legend(loc="center right",fontsize=10)
#plt.suptitle('Minimum Reaction Times for Animal Image Categorization')

#fig = plt.figure()
#for i in range(len(min_rt_subj['nonanimal'].keys())):
#    #for nonanimal
#    #graph successes
#    ax = fig.add_subplot(4,5,i+1)
#    plt.xlim(minrt,maxrt)
#    plt.ylim(y_min,y_max)
#    ylabs,ylocs = plt.yticks()
#    plt.yticks(np.arange(y_min,y_max,len(ylocs)),abs(ylabs).astype(int))
#    plt.title('Subject '+str(i))
#    plt.xlabel('RT (ms)')
#    plt.ylabel('Number of Images')
#    # find min rt
#    for bin in range(len(min_rt_hists['nonanimal'][i])):
#        if stats.binom_test([min_rt_hists['nonanimal'][i][bin],min_rt_hists['nonanimal_wrong'][i][bin]]) < .05:
#            min_rts['nonanimal'][i] = (bin*bin_width+(bin_width/2),min_rt_hists['nonanimal'][i][bin])
#            min_rts['nonanimal_p'][i] = stats.binom_test([min_rt_hists['nonanimal'][i][bin],min_rt_hists['animal_wrong'][i][bin]])
#            break
#    plt.bar(np.arange(minrt,maxrt,bin_width),min_rt_hists['nonanimal'][i], width=bin_width,hold=True, color='g',label='Correct')
#    plt.bar(np.arange(minrt,maxrt,bin_width),min_rt_hists['nonanimal_wrong'][i]*-1, width=bin_width,hold=True, color='r', label='Incorrect')
#    plt.plot((min_rts['nonanimal'][i][0],min_rts['nonanimal'][i][0]),(y_min,y_max),'b--',label='Min RT')
#    plt.legend(loc="center right",fontsize=10)
#plt.suptitle('Minimum Reaction Times for Non-Animal Image Categorization')         
    
#plt.close()
#plt.close()
         
         
#--- min rt hists ---
#animal

#y_min = 0
#y_max = 8
#plt.hist([i[0] for i in min_rts['animal'].values()])
#plt.xlim(minrt,maxrt)
#plt.ylim(y_min,y_max)        
#plt.xlabel('RT (ms)')
#plt.ylabel('Number of Images')
#plt.title('Animal Image Minimum Response Time Distribution (n = 20)')      
       
#nonanimal
#plt.figure()
#y_min = 0
#y_max = 8
#plt.hist([i[0] for i in min_rts['nonanimal'].values()])
#plt.xlim(minrt,maxrt)
#plt.ylim(y_min,y_max)        
#plt.xlabel('RT (ms)')
#plt.ylabel('Number of Images')
#plt.title('Non-Animal Image Minimum Response Time Distribution (n = 20)') 

#--- check model agreement against human results, "animal decision"
#for caffe fc7
#a = []
#layer = 'fc6'
#model = 'VGG19'
#modeldata = np.load('/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_fc6_svmNC-4_setb50k_0-19/00500.npz')
#model_comp['animal'][model+'_'+layer] = np.empty([1,animal_ind])
#model_comp['nonanimal'][model+'_'+layer] = np.empty([1,nonanimal_ind])
#for ind in range(len(modeldata['source_filenames'])):
#    impath = modeldata['source_filenames'][ind]
#    imname = impath.split('/')[-1]
#    a.append(modeldata['true_labels'][ind] == modeldata['pred_labels'][ind])
#    # if image was tested
#    if imname in im2key.keys():
#        model_comp[im2lab[imname]][model+'_'+layer][0][im2key[imname]] = modeldata['hyper_dist'][ind]
## plot human vs. caffe fc7
#ax = plt.gca()
##plt.ylim((-4.5,4.5)) #fc7
#plt.xlim((-10,110)) 
##plt.ylim((-45,40)) #conv4
##plt.ylim((-150,150)) #conv1
##plt.ylim((-3,5)) #VGG19 fc7
#plt.ylim((-400,300)) #VGG19 conv4_4
#xmin,xmax = ax.get_xlim()
#ymin,ymax = ax.get_ylim()
#plt.plot([xmin,xmax],[0,0], c='k', hold=True)
#plt.plot([0,0],[ymin,ymax], c='k', hold=True)
#plt.scatter(model_comp['animal']['human_acc'][0],model_comp['animal'][model+'_'+layer][0],c='b',label='Animal',hold=True)
#plt.scatter(100-model_comp['nonanimal']['human_acc'][0],model_comp['nonanimal'][model+'_'+layer][0],c='g',label='Non-Animal',hold=True)
#plt.plot([50,50],[ymin,ymax], c='r',ls='dashed', hold=True, label='Chance')
##relabel axis
#x_min = 0
#x_max = 101
#plt.xticks(np.arange(x_min,x_max,50))
#xlabs,xlocs = plt.xticks()
#plt.xticks(np.arange(x_min,x_max,50),abs(xlabs).astype(int))
#plt.legend(loc="lower right")
#plt.title('Fitting Human to VGG19 '+layer+' "Animal" Decisions')
#plt.xlabel('"Animal" Decision Frequency (%)')
#plt.ylabel('Distance from Hyperplane')
#
## Calculate correlation and significance 
#human_res = np.concatenate((model_comp['animal']['human_acc'][0],100-model_comp['nonanimal']['human_acc'][0]))
#model_res = np.concatenate((model_comp['animal'][model+'_'+layer][0],model_comp['nonanimal'][model+'_'+layer][0]))
#corr, p_val = stats.kendalltau(human_res,model_res)
#print "Animal Decision Corelation: %s, p-value: %s" % (str(corr),str(p_val))

# Summary stats
# Peak model vs. human accuracy 
#hum = np.concatenate((mrt_500_class,mrt_1000_class,mrt_2000_class))
#mod = pickle.load(open('temp_acc.p','rb'))
#t, p = stats.ttest_ind(hum, mod, equal_var=False)
#print 'Human vs. Model Accuracy, p =:',p



class Error(Exception):
   """Base class for other exceptions"""
   pass

class NotEveryImageSeenAtThisRTError(Error):
   """Raised when an image from the set wasn't seen at a given response time"""
   pass

def plot_corr_correct(inputfn, model, layer, corr, by_mrt,shouldCombine,plotCorrAcc):
    modeldata = np.load(inputfn)
    model_comp['animal'][model+'_'+layer] = np.ones([1,animal_ind])
    model_comp['nonanimal'][model+'_'+layer] = np.ones([1,nonanimal_ind])
    for index in range(len(modeldata['source_filenames'])):
        impath = modeldata['source_filenames'][index]
        imname = impath.split('/')[-1]+'.jpg'
        # if image was tested
        if imname in im2key.keys():
#            pred1 = np.exp(modeldata['logs'][index][0])
#            pred2 = np.exp(modeldata['logs'][index][1])
#            if modeldata['pred_labels'][index] == modeldata['true_labels'][index]:
#                to_use = max(pred1,pred2)
#            else:
#                to_use = min(pred1,pred2)
#            
            
#            model_comp[im2lab[imname]][model+'_'+layer][0][im2key[imname]] = to_use #modeldata['hyper_dist'][index]
            model_comp[im2lab[imname]][model+'_'+layer][0][im2key[imname]] = modeldata['hyper_dist'][index]
    ax = plt.gca()
    plt.xlim((-10,110))
    if model == 'caffe':
        mname = 'AlexNet'
        if layer == 'fc7':
            plt.ylim((-3.5,4.5)) #fc7
            ind = 6
        elif layer == 'fc6':
            plt.ylim((-50,55)) #fc6
            ind = 5
        elif layer == 'conv5':
            plt.ylim((-30,60)) #conv5
            ind = 4
        elif layer == 'conv4':
            plt.ylim((-30,40)) #conv4
            ind = 3
        elif layer == 'conv3':
            plt.ylim((-90,100)) #conv3
            ind = 2
        elif layer == 'conv2': 
            plt.ylim((-130,130)) #conv2
            ind = 1
        else:
            plt.ylim((-150,150)) #conv1
            ind = 0
    elif model == 'VGG16':
        mname = model
        if layer == 'fc7ex':
#            plt.ylim((-5,5)) #VGG16 fc7
            plt.ylim((-0.1,1.1))
            ind = 14
        elif layer == 'fc6ex':
            plt.ylim((-10,15)) #VGG16 fc6
            ind = 13
        elif layer == 'conv5_3ex':
            plt.ylim((-60,50)) #VGG16 conv5_3
            ind = 12
        elif layer == 'conv5_2ex':
            plt.ylim((-100,140)) #VGG16 conv5_2
            ind = 11
        elif layer == 'conv5_1ex':
#           plt.ylim((-150,250)) #VGG16 conv5_1
            plt.ylim((-0.1,1.1))
            ind = 10
        elif layer == 'conv4_3ex':
#           plt.ylim((-150,250)) #VGG16 conv4_3
            plt.ylim((-0.1,1.1))
            ind = 9
        elif layer == 'conv4_2ex':
            plt.ylim((-250, 500)) #VGG16 conv4_2
            ind = 8
        elif layer == 'conv4_1ex':
            plt.ylim((-450,600)) #VGG16 conv4_1
            ind = 7
        elif layer == 'conv3_3ex':
            plt.ylim((-450,800)) #VGG16 conv3_3
            ind = 6
        elif layer == 'conv3_2ex':
            plt.ylim((-800,1000)) #VGG16 conv3_2
            ind = 5
        elif layer == 'conv3_1ex':
            plt.ylim((-650,1000)) #VGG16 conv3_1
            ind = 4
        elif layer == 'conv2_2ex':
            plt.ylim((-650,800)) #VGG16 conv2_2
            ind = 3
        elif layer == 'conv2_1ex':
            plt.ylim((-600,600)) #VGG16 conv2_1
            ind = 2
        elif layer == 'conv1_2ex':
            plt.ylim((-450,500)) #VGG16 conv1_2
            ind = 1
        else:
            #plt.ylim((-80,80)) #VGG16 conv1_1
            plt.ylim((-.1,1.1)) 
            ind = 0
    elif model == 'gist':
        mname = "Gist"
        plt.ylim((-.2,.2)) #VGG16 conv1_1
        ind = 0
    else:
        mname = model
        if layer == 'fc7':
            plt.ylim((-3,5)) #VGG19 fc7
            ind = 17
        elif layer == 'fc6':
            plt.ylim((-10,15)) #VGG19 fc6
            ind = 16
        elif layer == 'conv5_4':
            plt.ylim((-10,45)) #VGG19 conv5_4
            ind = 15
        elif layer == 'conv5_3':
            plt.ylim((-20,95)) #VGG19 conv5_3
            ind = 14
        elif layer == 'conv5_2':
            plt.ylim((-150,350)) #VGG19 conv5_2
            ind = 13
        elif layer == 'conv5_1':
            plt.ylim((-150,350)) #VGG19 conv5_1
            ind = 12
        elif layer == 'conv4_4':
            plt.ylim((-400,350)) #VGG19 conv4_4
            ind = 11
        elif layer == 'conv4_3':
            plt.ylim((-600,900)) #VGG19 conv4_3
            ind = 10
        elif layer == 'conv4_2':
            plt.ylim((-1000,1400)) #VGG19 conv4_2
            ind = 9
        elif layer == 'conv4_1':
            plt.ylim((-1200,1800)) #VGG19 conv4_1
            ind = 8
        elif layer == 'conv3_4':
            plt.ylim((-1100,2000)) #VGG19 conv3_4
            ind = 7
        elif layer == 'conv3_3':
            plt.ylim((-550,700)) #VGG19 conv3_3
            ind = 6
        elif layer == 'conv3_2':
            plt.ylim((-400,500)) #VGG19 conv3_2
            ind = 5
        elif layer == 'conv3_1':
            plt.ylim((-400,600)) #VGG19 conv3_1
            ind = 4
        elif layer == 'conv2_2':
            plt.ylim((-400,500)) #VGG19 conv2_2
            ind = 3
        elif layer == 'conv2_1':
            plt.ylim((-400,500)) #VGG19 conv2_1
            ind = 2
        elif layer == 'conv1_2':
            plt.ylim((-300,300)) #VGG19 conv1_2
            ind = 1
        else:
            plt.ylim((-80,80)) #VGG19 conv1_1
            ind = 0
    if not(by_mrt):
        if not(plotCorrAcc):       
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            plt.plot([xmin,xmax],[0,0], c='k', hold=True)
            plt.plot([0,0],[ymin,ymax], c='k', hold=True)
            plt.scatter(model_comp['animal']['human_acc'][0],model_comp['animal'][model+'_'+layer][0],c='b',label='Animal',hold=True)
#            plt.scatter(model_comp['nonanimal']['human_acc'][0],model_comp['nonanimal'][model+'_'+layer][0],c='g',label='Non-Animal',hold=True)
            plt.scatter(model_comp['nonanimal']['human_acc'][0],-1*model_comp['nonanimal'][model+'_'+layer][0],c='g',label='Non-Animal',hold=True)
            plt.plot([50,50],[ymin,ymax], c='r',ls='dashed', hold=True, label='Chance')
            #relabel axis
            x_min = 0
            x_max = 101
            plt.xticks(np.arange(x_min,x_max,50))
            xlabs,xlocs = plt.xticks()
            plt.xticks(np.arange(x_min,x_max,50),abs(xlabs).astype(int))
            plt.legend(loc="upper left")
            if model == 'VGG16':
                layer1 = ' '+layer[0:len(layer)-2]
            elif model == 'gist':
                layer1 = ''
            else:
                layer1 = layer
            plt.title('Fitting Human to '+mname+layer1+' Accuracy')
            plt.xlabel('Human Accuracy (%)')
            plt.ylabel('Distance from Hyperplane')
        
        # Calculate correlation and significance 
        human_res = np.concatenate((model_comp['animal']['human_acc'][0],model_comp['nonanimal']['human_acc'][0]))
        model_res = np.concatenate((model_comp['animal'][model+'_'+layer][0],-1*model_comp['nonanimal'][model+'_'+layer][0]))
#        model_res = np.concatenate((model_comp['animal'][model+'_'+layer][0],model_comp['nonanimal'][model+'_'+layer][0]))
        if corr == "Spearman's rho":
            corrl, p_val = stats.spearmanr(human_res,model_res)
        elif corr == "Pearson's r":
            corrl, p_val = stats.pearsonr(human_res,model_res)
        else:
            corrl, p_val = stats.kendalltau(human_res,model_res)
        print "Correctness Corelation: %s, p-value: %s" % (str(corrl),str(p_val))
        if not(plotCorrAcc): 
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
            plt.figtext(0.5, 0.05, 'Correlation: %.3f (%s), p-value: %.2E' %(corrl,corr,decimal.Decimal(str(p_val))),horizontalalignment='center')
        else:
            acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels']))/float(len(modeldata['pred_labels']))
            if layer == 'fc6ex':
                temp_acc = (modeldata['pred_labels'] == modeldata['true_labels'])
                pickle.dump(temp_acc,open('temp_acc.p','wb'))
            
            print 'Accuracy: %.4f' %(acc)
            return (ind,corrl,acc)
    else:
        # Sepparate by mrt
        if not(shouldCombine):
            plt.close()
            max_rts = ['500','1000','2000']
            for mrt in max_rts:
                model_comp['animal']['human_acc_'+mrt] = np.empty([1,nonanimal_ind])*-500
                model_comp['nonanimal']['human_acc_'+mrt] = np.empty([1,nonanimal_ind])*-500
                for index in range(len(modeldata['source_filenames'])):
                    impath = modeldata['source_filenames'][index]
                    imname = impath.split('/')[-1]
                    # if image was tested
                    if imname in im2key.keys():
                        model_comp[im2lab[imname]]['human_acc_'+mrt][0][im2key[imname]] = float(np.mean(acc_by_im[imname.split('.')[0]][mrt]))*100        
            row = 1
            col = 3
        else:
            plt.close()
            max_rts = ['500','1000','2000']
            temp = {}
            for mrt in max_rts:
                if (mrt != '500'):
                    key ='1000-2000'
                    if not('human_acc_1000-2000' in model_comp['animal'].keys()):
                        model_comp['animal']['human_acc_'+key] = np.ones([1,nonanimal_ind])*-500
                        model_comp['nonanimal']['human_acc_'+key] = np.ones([1,nonanimal_ind])*-500
                        
                else:
                    key = mrt
                    model_comp['animal']['human_acc_'+mrt] = np.ones([1,nonanimal_ind])*-500
                    model_comp['nonanimal']['human_acc_'+mrt] = np.ones([1,nonanimal_ind])*-500
                for index in range(len(modeldata['source_filenames'])):
                    impath = modeldata['source_filenames'][index]
                    imname = impath.split('/')[-1]
                    # if image was tested
                    if imname in im2key.keys():
                        if key == '500':
                            model_comp[im2lab[imname]]['human_acc_'+key][0][im2key[imname]] = float(np.mean(acc_by_im[imname.split('.')[0]][mrt]))*100
                        else:
                            # first time image occurs
                            if not(im2lab[imname]+'human_acc_'+key+str(im2key[imname]) in temp.keys()):
                                temp[im2lab[imname]+'human_acc_'+key+str(im2key[imname])] = acc_by_im[imname.split('.')[0]][mrt]
                                model_comp[im2lab[imname]]['human_acc_'+key][0][im2key[imname]] = -501
                                #second time average
                            else:
                                model_comp[im2lab[imname]]['human_acc_'+key][0][im2key[imname]] = float(np.sum(acc_by_im[imname.split('.')[0]][mrt])+np.sum(temp[im2lab[imname]+'human_acc_'+key+str(im2key[imname])]))/float(len(acc_by_im[imname.split('.')[0]][mrt])+len(temp[im2lab[imname]+'human_acc_'+key+str(im2key[imname])]))*100
            row = 1
            col = 2
            max_rts = ['500','1000-2000']
        bad_sets = []
        for rt in max_rts:
            if len(filter(lambda a: a != -500, model_comp['animal']['human_acc_'+rt][0][:])) != len(model_comp['animal']['human_acc_'+rt][0][:]):
                bad_sets.append(rt+' animal')
            if len(filter(lambda a: a != -500, model_comp['nonanimal']['human_acc_'+rt][0][:])) != len(model_comp['animal']['human_acc_'+rt][0][:]):
                bad_sets.append(rt+' nonanimal')
        if len(bad_sets) >0:
            print "Not every image was seen in ", bad_sets
            raise NotEveryImageSeenAtThisRTError
        if not(plotCorrAcc):
            f, subp =  plt.subplots(row, col, sharey='row',squeeze=False)
        count = 0
        corrs = [0 for i in range(len(max_rts))]
        for r in range(row):
            for c in range(col):
                if not(plotCorrAcc):
                    ymin,ymax = ax.get_ylim()
                    subp[r][c].axis([-10,110,ymin,ymax])
                    subp[r][c].plot([-10,110],[0,0], c='k') 
                    subp[r][c].plot([0,0],[ymin,ymax], c='k')
                    subp[r][c].scatter(model_comp['animal']['human_acc_'+max_rts[count]][0],model_comp['animal'][model+'_'+layer][0],c='b',label='Animal')
                    subp[r][c].scatter(model_comp['nonanimal']['human_acc_'+max_rts[count]][0],-1*model_comp['nonanimal'][model+'_'+layer][0],c='g',label='Non-Animal')
                    subp[r][c].plot([50,50],[ymin,ymax], c='r',ls='dashed', label='Chance')
                    subp[r][c].set_xlabel('Human Accuracy (%)')
                    subp[r][c].set_ylabel('Distance from Hyperplane')
                    subp[r][c].set_title(max_rts[count]+'ms Max Response Time')
                # Calculate correlation and significance
                human_res = np.concatenate((model_comp['animal']['human_acc_'+max_rts[count]][0],model_comp['nonanimal']['human_acc_'+max_rts[count]][0]))
                model_res = np.concatenate((model_comp['animal'][model+'_'+layer][0],-1*model_comp['nonanimal'][model+'_'+layer][0]))
                if corr == "Spearman's rho":
                    corrl, p_val = stats.spearmanr(human_res,model_res)
                elif corr == "Pearson's r":
                    corrl, p_val = stats.pearsonr(human_res,model_res)
                else:
                    corrl, p_val = stats.kendalltau(human_res,model_res)
                print "Correctness Corelation: %s, p-value: %s" % (str(corrl),str(p_val))
                if not(plotCorrAcc):
                    box = subp[r][c].get_position()
                    subp[r][c].set_position([box.x0, box.y0 + box.height * 0.15,box.width, box.height * 0.85])
                    f.text(box.x0+box.width/2,box.y0, 'Correlation: %.3f (%s), p-value: %.2E' %(corrl,corr,decimal.Decimal(str(p_val))), horizontalalignment='center') 
                else:
                    corrs[c] = corrl
                count +=1
        if not(plotCorrAcc):
            if model == 'VGG16':
                layer1 = ' '+layer[0:len(layer)-2]
            elif layer == 'gist':
                layer1 = ''
            else:
                layer1 = layer
            f.suptitle('Fitting Human to ' +mname+' '+layer1+' Accuracy for Varied Max Response Times')
        acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels']))/float(len(modeldata['pred_labels']))
        print 'Accuracy: %.2f' %(acc)
        return (ind,corrs,acc)

def get_ind(layer):
    if layer == 'VGG16_fc7ex':
        ind = 14
    elif layer == 'VGG16_fc6ex':
        ind = 13
    elif layer == 'VGG16_conv5_3ex':
        ind = 12
    elif layer == 'VGG16_conv5_2ex':
        ind = 11
    elif layer == 'VGG16_conv5_1ex':
        ind = 10
    elif layer == 'VGG16_conv4_3ex':
        ind = 9
    elif layer == 'VGG16_conv4_2ex':
        ind = 8
    elif layer == 'VGG16_conv4_1ex':
        ind = 7
    elif layer == 'VGG16_conv3_3ex':
        ind = 6
    elif layer == 'VGG16_conv3_2ex':
        ind = 5
    elif layer == 'VGG16_conv3_1ex':
        ind = 4
    elif layer == 'VGG16_conv2_2ex':
        ind = 3
    elif layer == 'VGG16_conv2_1ex':
        ind = 2
    elif layer == 'VGG16_conv1_2ex':
        ind = 1
    else:
        ind = 0   
    return ind


def bootstrap(n_samples,alpha,feats,im2lab, n_subj):
    # Calculate stats per image for sample populataion 
    model_comp = {} # data for model comparison
    model_comp['animal'] = {}
    model_comp['nonanimal'] = {}
    im2key = {} # maps image to index in model comp array
    animal_ind = 0;
    nonanimal_ind =0;
    accs = []
    # store image names with index in dict to hold model comparisons
    for im in im2lab.keys():
        if im2lab[im] == 'animal':
            im2key[im] = animal_ind
            animal_ind +=1
        else:
            im2key[im] = nonanimal_ind
            nonanimal_ind +=1
    for feat in feats:
        # calculate features
        modeldata = np.load(feats[feat])
        model_comp['animal'][feat] = np.ones([1,animal_ind])
        model_comp['nonanimal'][feat] = np.ones([1,nonanimal_ind])
        for index in range(len(modeldata['source_filenames'])):
            impath = modeldata['source_filenames'][index]
            imname = impath.split('/')[-1]+'.jpg'
            # if image was tested
            if imname in im2key.keys():
                model_comp[im2lab[imname]][feat][0][im2key[imname]] = modeldata['hyper_dist'][index]                  
    inds = np.random.randint(0,n_subj,(n_samples,n_subj))
    sample_corrs = np.ones((n_samples,len(feats.keys())))
    for draw in range(n_samples):
        acc = []
        # Calculate stats per image for sample populataion 
        model_comp['animal']['source_images'] = np.empty([animal_ind,1],dtype='S20')
        model_comp['nonanimal']['source_images'] = np.empty([nonanimal_ind,1],dtype='S20')
        for im in im2key.keys():
            model_comp[im2lab[im]]['source_images'][im2key[im]][0] = im
        model_comp['animal']['human_acc'] = np.empty([1,animal_ind]) #arrays to hold human accuracy data for model comparison      
        model_comp['nonanimal']['human_acc'] = np.empty([1,nonanimal_ind])
        im_keys = {}
        for participant in inds[draw]:
            for img in im_by_subj[participant].keys():
                if not(img in im_keys.keys()):
                    im_keys[img] = img
        # image stats
        for im in im_keys.keys():
            results = []
            for participant in inds[draw]:
                if im in im_by_subj[participant].keys():
                    results.append(im_by_subj[participant][im])
                    acc.append(im_by_subj[participant][im])
            score = np.mean(results)
            if im2lab[im+'.jpg'] == 'animal':
                model_comp['animal']['human_acc'][0][im2key[im+'.jpg']] = float(score*100)
            else:
                model_comp['nonanimal']['human_acc'][0][im2key[im+'.jpg']] = float(score*100)
            
        accs.append(np.mean(acc))   
        for feat in feats:
                
            # Calculate correlation and significance
            human_res = np.concatenate((model_comp['animal']['human_acc'][0],model_comp['nonanimal']['human_acc'][0]))
            model_res = np.concatenate((model_comp['animal'][feat][0],-1*model_comp['nonanimal'][feat][0]))
            corrl, p_val = stats.spearmanr(human_res,model_res)
            sample_corrs[draw][get_ind(feat)] = corrl
                
        if (draw)%100 ==0:
            if draw == 0:
                print "Starting sampling..."
            else:
                print 'Completed %d out of %d samples' %(draw,n_samples)  
    CIs = np.ones([2,len(feats.keys())])
    for i in range(np.size(sample_corrs,1)):
        curr = np.sort(sample_corrs[:,i])
        CIs[0][i] = curr[int((alpha/2.0)*n_samples)]
        CIs[1][i] = curr[int(1-(alpha/2.0)*n_samples)]
    accstats = np.sort(accs)
    berr = accstats[int((alpha/2.0)*len(accs))]
    terr = accstats[int(1-(alpha/2.0)*len(accs))]
    return (CIs,accs,[terr,berr])

        
def overviewPlot(model, corr, by_mrt,shouldCombine, feats, total_acc,abstract, n_subj):
    if model =='VGG19':
        n_layers = 18
    elif model == 'VGG16':
        n_layers = 15
    else:
        n_layers = 7
    if (by_mrt):
        if not(shouldCombine):
            corrs = np.ones((3,n_layers))
            accs = np.ones((1,n_layers))
            for key in feats.keys():
                plt.close()
                infn = feats[key]
                model = key.split('_')[0]
                if len(key.split('_')) == 2:  
                    layer = key.split('_')[1]
                else:
                    layer = key.split('_')[1]+'_'+key.split('_')[2]
                ind,cr,ac = plot_corr_correct(infn, model, layer, corr, by_mrt, shouldCombine, True)
                corrs[0][ind] = cr[0]
                corrs[1][ind] = cr[1]
                corrs[2][ind] = cr[2]
                accs[0][ind] = ac
            plt.xlim((0,n_layers-1))
            plt.ylim((0,1))
            plt.plot(xrange(n_layers),accs[0][:],c='k',hold=True,label='Accuracy')
            plt.plot(xrange(n_layers),corrs[0][:],c='b',hold=True,label='Correlation (500ms)')
            plt.plot(xrange(n_layers),corrs[1][:],c='g',hold=True,label='Correlation (1000ms)')
            plt.plot(xrange(n_layers),corrs[2][:],c='r',label='Correlation (2000ms)')
            plt.ylabel(corr+' Correlation/Layer Accuracy')
            plt.xlabel('Layer (Increasing Complexity)')
            if model == 'caffe':
                model = 'AlexNet'
            if model == 'VGG16ex':
                model = 'VGG16'
            plt.title(model+' Model Accuracy and Correlation with Human Decisions by Max RT')
            plt.legend(loc="upper left")
        else:
            corrs = np.ones((2,n_layers))
            accs = np.ones((1,n_layers))
            for key in feats.keys():
                plt.close()
                infn = feats[key]
                model = key.split('_')[0]
                if len(key.split('_')) == 2:  
                    layer = key.split('_')[1]
                else:
                    layer = key.split('_')[1]+'_'+key.split('_')[2]
                ind,cr,ac = plot_corr_correct(infn, model, layer, corr, by_mrt, shouldCombine, True)
                corrs[0][ind] = cr[0]
                corrs[1][ind] = cr[1]
                accs[0][ind] = ac
            plt.xlim((0,n_layers-1))
            plt.ylim((0,1))
            plt.plot(xrange(n_layers),accs[0][:],c='k',hold=True,label='Accuracy')
            plt.plot(xrange(n_layers),corrs[0][:],c='b',hold=True,label='Correlation (500ms)')
            plt.plot(xrange(n_layers),corrs[1][:],c='g',label='Correlation (1000-2000ms)')
            plt.ylabel(corr+' Correlation/Layer Accuracy')
            plt.xlabel('Layer (Increasing Complexity)')
            if model == 'caffe':
                model = 'AlexNet'
            if model == 'gist':
                model = 'Gist'
            plt.title(model+' Model Accuracy and Correlation with Human Decisions by Max RT')
            plt.legend(loc="upper left")
    else: 
        if not(abstract):
            corrs = np.ones((1,n_layers))
            accs = np.ones((1,n_layers))
            for key in feats.keys():
                infn = feats[key]
                model = key.split('_')[0]
                if len(key.split('_')) == 2:  
                    layer = key.split('_')[1]
                else:
                    layer = key.split('_')[1]+'_'+key.split('_')[2]
                ind,cr,ac = plot_corr_correct(infn, model, layer, corr, by_mrt, shouldCombine, True)
                corrs[0][ind] = cr
                accs[0][ind] = ac
            plt.xlim((0,n_layers-1))
            plt.ylim((0,1))
            plt.plot(xrange(n_layers),accs[0][:],c='k',hold=True,label='Accuracy')
            plt.plot(xrange(n_layers),corrs[0][:],c='b',label='Correlation')
            plt.ylabel(corr+' Correlation/Layer Accuracy')
            plt.xlabel('Layer (Increasing Complexity)')
            if model == 'caffe':
                model = 'AlexNet'
            plt.title(model+' Model Accuracy and Correlation with Human Decisions')
            plt.legend(loc="upper left")
        else:
            fig, ax1 = plt.subplots()
            corrs = np.ones((1,n_layers))
            accs = np.ones((1,n_layers))
            for key in feats.keys():
                infn = feats[key]
                model = key.split('_')[0]
                if len(key.split('_')) == 2:  
                    layer = key.split('_')[1]
                else:
                    layer = key.split('_')[1]+'_'+key.split('_')[2]
                ind,cr,ac = plot_corr_correct(infn, model, layer, corr, by_mrt, shouldCombine, True)
                corrs[0][ind] = cr
                accs[0][ind] = ac
            
            #bootstrap CIs
            confs, hacc, errs = bootstrap(1000,.05,feats,im2lab,n_subj)
            pickle.dump(confs,open('c_conf_vals.p','wb'))
            pickle.dump(hacc,open('c_hacc.p','wb'))
            pickle.dump(errs,open('c_errs.p','wb'))
            #pickle.dump(accs,open('accs.p','wb'))
            #hacc = pickle.load(open('hacc.p','rb'))
            #errs = pickle.load(open('errs.p','rb'))
            #confs = pickle.load(open('conf_vals.p','rb'))
            ax1.set_ylim([48,100])
            plt.xticks(rotation=70)
            #calculate human mean per-images accuarcy
            hum_im_acc = []
            for im in acc_by_im.keys():
                res = []
                for mrt in acc_by_im[im].keys():
                    res = np.concatenate((res,acc_by_im[im][mrt]))
                hum_im_acc.append(np.mean(res))
            ax1.plot([-0.5,n_layers-0.5],[int(np.mean(hum_im_acc)*100),int(np.mean(hum_im_acc)*100)],color='0.5',ls='--')
            ax1.plot([-0.5,n_layers-0.5],[int(errs[0]*100),int(errs[0]*100)],color='0.75',ls='--')
            ax1.plot([-0.5,n_layers-0.5],[int(errs[1]*100),int(errs[1]*100)],color='0.75',ls='--')
            ax1.plot(xrange(n_layers),accs[0][:]*100,'bo')
            ax1.set_ylabel('Accuracy (%)',color='b')
            p1_fit = np.polyfit(xrange(n_layers),accs[0][:]*100,2)
            p1_fn = np.poly1d(p1_fit)
            xs = np.linspace(0, n_layers-1)
            ax1.plot(xs,p1_fn(xs),'b')
            for tick in ax1.get_yticklabels():
                tick.set_color('b')        
    
            ax2 = ax1.twinx()
            
            ax2.set_ylim([-.1,1])
            ax2.plot(xrange(n_layers),corrs[0][:],'ro')
            ax2.set_ylabel(corr+' Correlation', color='r')
            p2_fit = np.polyfit(xrange(n_layers),corrs[0][:],3)
            p2_fn = np.poly1d(p2_fit)
            xs = np.linspace(0, n_layers-1)
            ax2.plot(xs,p2_fn(xs),'r')
            ax2.text(1.5,.51, 'Human Accuracy',color='.5') 
            for tick in ax2.get_yticklabels():
                tick.set_color('r')
                
            p3_fit = np.polyfit(xrange(n_layers),confs[0][:],3)
            p3_fn = np.poly1d(p3_fit)
            xs = np.linspace(0, n_layers-1)
            ax2.plot(xs,p3_fn(xs),'#ffb3b3')
            p4_fit = np.polyfit(xrange(n_layers),confs[1][:],3)
            p4_fn = np.poly1d(p4_fit)
            xs = np.linspace(0, n_layers-1)
            ax2.plot(xs,p4_fn(xs),'#ffb3b3')
            
            ax2.set_xlim([-0.5, n_layers-0.5])
            plt.sca(ax2)
            plt.xticks(range(n_layers),['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3','fc6','fc7'])
            plt.sca(ax1)
            ax1.set_xlabel('Layer (Increasing Complexity)')
            if model == 'caffe':
                model = 'AlexNet'
            plt.title(model+' Model Accuracy and Correlation with Human Decisions')
            

def overview_double_boot(model, feats, corr, hacc_dict,n_subj, n_samples, alpha):
    if model =='VGG19':
        n_layers = 18
    elif model == 'VGG16':
        n_layers = 15
    else:
        n_layers = 7
    corrs = np.ones((2,n_layers))
    accs = np.ones((1,n_layers))
    fig, ax1 = plt.subplots()
    for key in feats.keys():
        plt.close()
        infn = feats[key]
        model = key.split('_')[0]
        if len(key.split('_')) == 2:  
            layer = key.split('_')[1]
        else:
            layer = key.split('_')[1]+'_'+key.split('_')[2]
        ind,cr,ac = plot_corr_correct(infn, model, layer, corr, True, True, True)
        corrs[0][ind] = cr[0]
        corrs[1][ind] = cr[1]
        accs[0][ind] = ac
    #bootstrap CIs
    # Calculate stats per image for sample populataion 
    model_comp = {} # data for model comparison
    model_comp['animal'] = {}
    model_comp['nonanimal'] = {}
    im2key = {} # maps image to index in model comp array
    animal_ind = 0;
    nonanimal_ind =0;
    modelaccs = accs
    accs = []
    # store image names with index in dict to hold model comparisons
    for im in im2lab.keys():
        if im2lab[im] == 'animal':
            im2key[im] = animal_ind
            animal_ind +=1
        else:
            im2key[im] = nonanimal_ind
            nonanimal_ind +=1
    for feat in feats:
        # calculate features
        modeldata = np.load(feats[feat])
        model_comp['animal'][feat] = np.ones([1,animal_ind])
        model_comp['nonanimal'][feat] = np.ones([1,nonanimal_ind])
        for index in range(len(modeldata['source_filenames'])):
            impath = modeldata['source_filenames'][index]
            imname = impath.split('/')[-1]
            # if image was tested
            if imname in im2key.keys():
                model_comp[im2lab[imname]][feat][0][im2key[imname]] = modeldata['hyper_dist'][index]                  
    inds = np.random.randint(0,n_subj,(n_samples,n_subj))
    sample_corrs = np.ones((n_samples,len(feats.keys())))
    mrts = ['500', '1000-2000']
    
    bothCIs = [] 
    bothaccs = []
    botherrs = []
    
    for mrt in mrts:
        for draw in range(n_samples):
            acc = []
            # Calculate stats per image for sample populataion 
            model_comp['animal']['source_images'] = np.empty([animal_ind,1],dtype='S20')
            model_comp['nonanimal']['source_images'] = np.empty([nonanimal_ind,1],dtype='S20')
            for im in im2key.keys():
                model_comp[im2lab[im]]['source_images'][im2key[im]][0] = im
            model_comp['animal']['human_acc'] = np.empty([1,animal_ind]) #arrays to hold human accuracy data for model comparison      
            model_comp['nonanimal']['human_acc'] = np.empty([1,nonanimal_ind])
            im_keys = {}
            for participant in inds[draw]:
                for img in im_by_subj[participant].keys():
                    if not(img in im_keys.keys()):
                        im_keys[img] = img
            # image stats
            for im in im_keys.keys():
                results = []
                for participant in inds[draw]:
                    if im in im_by_subj[participant].keys():
                        if im in hacc_dict[participant][mrt].keys():
                            results.append(hacc_dict[participant][mrt][im])
                        acc.append(im_by_subj[participant][im])
                score = np.mean(results)
                if im2lab[im+'.jpg'] == 'animal':
                    model_comp['animal']['human_acc'][0][im2key[im+'.jpg']] = float(score*100)
                else:
                    model_comp['nonanimal']['human_acc'][0][im2key[im+'.jpg']] = float(score*100)
                
            accs.append(np.mean(acc))   
            for feat in feats:
                # Calculate correlation and significance
                human_res = np.concatenate((model_comp['animal']['human_acc'][0],model_comp['nonanimal']['human_acc'][0]))
                model_res = np.concatenate((model_comp['animal'][feat][0],-1*model_comp['nonanimal'][feat][0]))
                corrl, p_val = stats.spearmanr(human_res,model_res)
                sample_corrs[draw][get_ind(feat)] = corrl
                    
            if (draw)%100 ==0:
                if draw == 0:
                    print "Starting sampling..."
                else:
                    print 'Completed %d out of %d samples' %(draw,n_samples)  
        CIs = np.ones([2,len(feats.keys())])
        for i in range(np.size(sample_corrs,1)):
            curr = np.sort(sample_corrs[:,i])
            CIs[0][i] = curr[int((alpha/2.0)*n_samples)]
            CIs[1][i] = curr[int(1-(alpha/2.0)*n_samples)]
        accstats = np.sort(accs)
        berr = accstats[int((alpha/2.0)*len(accs))]
        terr = accstats[int(1-(alpha/2.0)*len(accs))]
        bothCIs.append(CIs)
        bothaccs.append(accs)
        botherrs.append([terr,berr])
    # plot 
    ax1.set_ylim([48,100])
    ax1.XTickLabelRotation=70;
    errs = botherrs[0]
    #calculate human mean per-images accuarcy
    hum_im_acc = []
    for im in acc_by_im.keys():
        res = []
        for mrt in acc_by_im[im].keys():
            res = np.concatenate((res,acc_by_im[im][mrt]))
        hum_im_acc.append(np.mean(res))
    ax1.plot([-0.5,n_layers-0.5],[int(np.mean(hum_im_acc)*100),int(np.mean(hum_im_acc)*100)],color='0.5',ls='--')
    ax1.plot([-0.5,n_layers-0.5],[int(errs[0]*100),int(errs[0]*100)],color='0.75',ls='--')
    ax1.plot([-0.5,n_layers-0.5],[int(errs[1]*100),int(errs[1]*100)],color='0.75',ls='--')
    
    ax1.plot(xrange(n_layers),modelaccs[0]*100,'bo')
    
    ax1.set_ylabel('Accuracy (%)',color='b')
    p1_fit = np.polyfit(xrange(n_layers),modelaccs[0][:]*100,2)
    p1_fn = np.poly1d(p1_fit)
    xs = np.linspace(0, n_layers-1)
    ax1.plot(xs,p1_fn(xs),'b')
    for tick in ax1.get_yticklabels():
        tick.set_color('b')        
        
    ax2 = ax1.twinx()
    ax2.set_ylim([-.1,1])
                
    ax2.plot(xrange(n_layers),corrs[0][:],'ro',label='500ms')
    ax2.plot(xrange(n_layers),corrs[1][:],'r>',label='1000-2000ms')
    ax2.set_ylabel(corr+' Correlation', color='r')
    p2_fit = np.polyfit(xrange(n_layers),corrs[0][:],3)
    p2_fn = np.poly1d(p2_fit)
    p22_fit = np.polyfit(xrange(n_layers),corrs[1][:],3)
    p22_fn = np.poly1d(p22_fit)
    xs = np.linspace(0, n_layers-1)
    ax2.plot(xs,p2_fn(xs),'r')
    ax2.plot(xs,p22_fn(xs),'r')
                
                
                
    ax2.text(1.5,.6, 'Human Accuracy',color='.5') 
    for tick in ax2.get_yticklabels():
        tick.set_color('r')
                    
    
    ax2.set_xlim([-0.5, n_layers-0.5])
    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels(['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3','fc6','fc7'],rotation=70)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(0)
    ax1.set_xlabel('Layer (Increasing Complexity)')
    if model == 'caffe':
        model = 'AlexNet'
    ax1.set_title(model+' Model Accuracy and Correlation with Human Decisions')
    return fig

    
#input lists
corrs = ["Spearman's rho","Pearson's r", "Kendall's tau"]
caffe_feats = {}
VGG19_feats = {}
VGG16_feats = {}
gist_feats = {}
caffe_feats['caffe_fc7'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_fc7_svmNC-4_setb50k_0-19/00500.npz'
caffe_feats['caffe_fc6'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_fc6_svmNC-3_setb50k_0-19/00500.npz'
caffe_feats['caffe_conv5'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_conv5_svmNC-4_setb50k_0-19/00500.npz'
caffe_feats['caffe_conv4'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_conv4_svmNC-4_setb50k_0-19/00500.npz'
caffe_feats['caffe_conv3'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_conv3_svmNC-4_setb50k_0-19/00500.npz'
caffe_feats['caffe_conv2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_conv2_svmNC-4_setb50k_0-19/00500.npz'
caffe_feats['caffe_conv1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/caffe_conv1_svmNC-5_setb50k_0-19/00500.npz'

VGG16_feats['VGG16_fc7ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_fc7ex_svmNC-4_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_fc6ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_fc6ex_svmNC-4_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv5_3ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv5_3ex_svmNC-4_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv5_2ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv5_2ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv5_1ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv5_1ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv4_3ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv4_3ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv4_2ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv4_2ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv4_1ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv4_1ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv3_3ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv3_3ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv3_2ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv3_2ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv3_1ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv3_1ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv2_2ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv2_2ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv2_1ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv2_1ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv1_2ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv1_2ex_svmNC-5_set1603241729_0-15/1603241729.npz'
VGG16_feats['VGG16_conv1_1ex'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG16_conv1_1ex_svmNC-5_set1603241729_0-15/1603241729.npz'


VGG19_feats['VGG19_fc7'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_fc7_svmNC-4_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_fc6'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_fc6_svmNC-4_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv5_4'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv5_4_svmNC-4_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv5_3'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv5_3_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv5_2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv5_2_svmNC-4_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv5_1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv5_1_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv4_4'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv4_4_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv4_3'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv4_3_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv4_2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv4_2_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv4_1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv4_1_svmNC-4_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv3_4'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv3_4_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv3_3'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv3_3_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv3_2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv3_2_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv3_1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv3_1_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv2_2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv2_2_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv2_1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv2_1_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv1_2'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv1_2_svmNC-5_setb50k_0-19/00500.npz'
VGG19_feats['VGG19_conv1_1'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/VGG19_conv1_1_svmNC-5_setb50k_0-19/00500.npz'

gist_feats['gist'] = '/home/jcader/Thesis/dmtcat_analysis/predictions/gist_svmNC-3_setb50k_0-19/00500.npz'

#plot_corr_correct(VGG19_feats['VGG19_conv2_1'], 'VGG19', 'conv2_1',  "Pearson's r",True,True)

# AlexNet Correlation Plots
#for key in caffe_feats.keys():
#    infn = caffe_feats[key]
#    model = key.split('_')[0]
#    layer = key.split('_')[1]
#    corr = corrs[2]
#    plt.figure()
#    plot_corr_correct(infn, model, layer, corr,True, True, False)

# AlexNet Overview Plots
#overviewPlot('caffe', corrs[2], True,True,caffe_feats)

# VGG16 Correlation Plots

#for key in VGG16_feats.keys():
#    infn = VGG16_feats[key]
#    model = key.split('_')[0]
#    if len(key.split('_'))==2:  
#        layer = key.split('_')[1]+'ex'
#    else:
#        layer = key.split('_')[1]+'_'+key.split('_')[2]+'ex'
#    corr = corrs[0]
#    plt.figure()
#    plot_corr_correct(infn, 'VGG16', layer, corr, False, False, False)

#key = 'VGG16_fc7ex'
#infn = VGG16_feats[key]
#model = key.split('_')[0]
#if len(key.split('_'))==2:  
#    layer = key.split('_')[1]
#else:
#    layer = key.split('_')[1]+'_'+key.split('_')[2]
#corr = corrs[0]
#plot_corr_correct(infn, 'VGG16', layer, corr, False, False, False)
## VGG16 Overview Plots
#overviewPlot('VGG16', corrs[0], True, True,VGG16_feats,total_acc,True,77)


# VGG19 Correlation Plots

#for key in VGG19_feats.keys():
#    infn = VGG19_feats[key]
#    model = key.split('_')[0]
#    if len(key.split('_'))==2:  
#        layer = key.split('_')[1]
#    else:
#        layer = key.split('_')[1]+'_'+key.split('_')[2]
#    corr = corrs[2]
#    plt.figure()
#    plot_corr_correct(infn, model, layer, corr, False, False, False)

# VGG19 Overview Plots
#overviewPlot('VGG19', corrs[0], False,False,VGG16_feats, total_acc)

#for key in gist_feats.keys():
#    infn = gist_feats[key]
#    model = key
#    layer = key
#    corr = corrs[0]
#    plt.figure()
#    plot_corr_correct(infn, model, layer, corr, True, False, False)

#fig = overview_double_boot('VGG16', VGG16_feats, corrs[0], data_by_subj_by_rt,20, 1000, .05)
P.show
plt.show()