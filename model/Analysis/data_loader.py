#!/usr/bin/env python
# Load experimental and model data

import sqlite3
import json
import numpy as np
from rapid_categorization.model import util
from results_key import label_results
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from statsmodels.stats.multicomp import MultiComparison
from scipy import stats
import warnings

class Data:
    def __init__(self):
        self.im2lab = {}
        self.model_comp = {} # data for model comparison
        self.im2key = {}  # maps image to index in model comp array
        self.im2path = {}  # maps image filename to containing folder
        self.animal_ind = 0
        self.nonanimal_ind = 0
        self.trials_by_img = {}  # trial data indexed by image
        self.data_by_img = {}  # result data averaged over trials by image
        self.data_by_max_rt = {} #results by max response time
        self.data_by_max_rt['500'] =[[]for i in range(50)]
        self.data_by_max_rt['1000'] =[[]for i in range(50)]
        self.data_by_max_rt['2000'] =[[]for i in range(50)]
        self.responses_by_max_rt = {} #results by max response time
        self.responses_by_max_rt['500'] =[[]for i in range(50)]
        self.responses_by_max_rt['1000'] =[[]for i in range(50)]
        self.responses_by_max_rt['2000'] =[[]for i in range(50)]
        self.responses_by_max_rt['2000'] =[[]for i in range(50)]
        self.acc_by_im = {}
        self.avg_acc_by_im = {}
        self.total_acc = []
        self.total_time = []
        self.min_rt_subj = {}
        self.min_rt_subj['animal'] = {}
        self.min_rt_subj['nonanimal'] = {}
        self.min_rt_subj['animal_wrong'] = {}
        self.min_rt_subj['nonanimal_wrong'] = {}
        self.im_by_subj= {}
        # Track failure rates
        self.failures = [[0, 0], [0, 0], [0, 0]]
        # rt hists by max_rt
        self.rts = ['500', '1000', '2000']
        self.mrt_500 = []
        self.mrt_1000 = []
        self.mrt_2000 = []
        self.mrt_500_25 = []
        self.mrt_1000_25 = []
        self.mrt_2000_25 = []
        self.mrt_500_class = []
        self.mrt_1000_class = []
        self.mrt_2000_class = []
        self.mrt_500_25_class = []
        self.mrt_1000_25_class = []
        self.mrt_2000_25_class = []
        self.mrt_500_avgrt = []
        self.mrt_1000_avgrt = []
        self.mrt_2000_avgrt = []
        self.mrt_500_avgclass = []
        self.mrt_1000_avgclass = []
        self.mrt_2000_avgclass = []
        self.fail_500 = None
        self.fail_1000 = None
        self.fail_2000 = None
        self.hum_im_acc = []
        self.loaded_sets = []
        self.model_corrs = {}
        self.model_accs = {}
        self.model_corrs_bootstrapped = {}
        self.human_acc_bootstrapped = []
        self.human_acc_by_image_bootstrapped = []
        self.corr_type = 'None'
        self.n_subjects = 0
        self.n_trials = 0
        self.n_timeouts = 0
        self.ignore_timeouts = True
        self.correct_rts = []

    def load(self, experiment_id):
        print 'Loading experiment %d' % experiment_id
        set_index, set_name = util.get_experiment_imageset(experiment_id)
        self.load_ground_truth(set_index, set_name)
        self.load_participants(experiment_id)

    def load_ground_truth(self, set_index, set_name):
        imageset_filename = util.get_imageset_filename(set_index, set_name)
        # Generate ground truth key
        self.im2lab.update(label_results(imageset_filename))
        self.loaded_sets += [(set_index, set_name)]

    def load_participant_json(self, experiment_id, verbose=True):
        exp_filename = util.get_experiment_db_filename(experiment_id)
        r = sqlite3.connect(exp_filename).cursor().execute(
            "SELECT workerid,beginhit,status,datastring FROM placecat WHERE status in (3,4) AND NOT datastring==''").fetchall()
        if verbose: print "%d participants found in file %s." % (len(r), exp_filename)
        return r

    def load_participants(self, experiment_id):
        exp_filename = util.get_experiment_db_filename(experiment_id)
        r = self.load_participant_json(experiment_id, verbose=True)
        for i_subj in range(0,len(r)):
            data = json.loads(r[i_subj][3])
            self.load_subject(data)

    def load_im2path(self, experiment_ids):
        self.im2path = {}
        for experiment_id in experiment_ids:
            set_index, set_name = util.get_experiment_imageset(experiment_id)
            imageset_filename = util.get_imageset_filename(set_index, set_name)
            im_names = label_results(imageset_filename)
            im_root = util.get_input_image_root(set_index, set_name)
            for k in im_names.iterkeys():
                self.im2path[k] = im_root

    def load_multi(self, experiment_ids, bootstrap_size=None):
        r = []
        for experiment_id in experiment_ids:
            set_index, set_name = util.get_experiment_imageset(experiment_id)
            self.load_ground_truth(set_index, set_name)
            r += self.load_participant_json(experiment_id, verbose=(bootstrap_size is None))
        if bootstrap_size is not None:
            idcs = np.random.choice(range(len(r)), size=bootstrap_size, replace=True)
            r = [r[i] for i in idcs]
        self.subject_ids = np.unique([subj[0] for subj in r])
        if bootstrap_size is None:
            print '%d unique subjects' % (len(self.subject_ids))
        for subj in r:
            #print "subj %d(%d trials) = %s" % (i_subj, len(trials), self.r[i_subj][0])
            self.load_subject(json.loads(subj[3]))
        if bootstrap_size is None:
            print '%d trials, %d timeouts (%.1f%%)' % (self.n_trials, self.n_timeouts, float(self.n_timeouts) / self.n_trials * 100)

    def bootstrap(self, experiment_ids, model_names, classifier_type, train_batches, corr_type, adjust_corr_for_true_label, bootstrap_count, bootstrap_size):
        # Bootstrap model correlations and human accuracy values
        model_corrs_bootstrapped = {}
        human_acc_bootstrapped = []
        human_acc_by_image_bootstrapped = []
        for i_boot in xrange(bootstrap_count):
            print 'bootstrap iter %d of %d...' % (i_boot+1, bootstrap_count)
            bdata = Data()
            bdata.load_multi(experiment_ids, bootstrap_size=bootstrap_size)
            bdata.eval_participants()
            for model_name in model_names:
                bdata.load_model_data(model_name, classifier_type, train_batches)
                bdata.calc_model_correlation(model_name, corr_type, adjust_corr_for_true_label, verbose=False)
            for k,corr in bdata.model_corrs.iteritems():
                c = corr[0]
                if i_boot==0:
                    model_corrs_bootstrapped[k] = []
                model_corrs_bootstrapped[k] += [c]
            human_acc_bootstrapped += [np.mean(bdata.hum_im_acc)]
            human_acc_by_image_bootstrapped += [bdata.hum_im_acc]
        self.model_corrs_bootstrapped = model_corrs_bootstrapped
        self.human_acc_bootstrapped = human_acc_bootstrapped
        self.human_acc_by_image_bootstrapped = human_acc_by_image_bootstrapped

    def load_subject(self, data):
        i_subj = self.n_subjects
        self.n_subjects += 1
        self.im_by_subj[i_subj] = {}
        # Get experimental trials.
        alltrials = data['data']
        # Ignore training trials
        exptrials = alltrials[10:]
        # Ignore pause screens
        trials = [t['trialdata'] for t in exptrials if t['trialdata']['trial_type'] == 's2stim']

        # Store data by subj
        self.min_rt_subj['animal'][i_subj] = []
        self.min_rt_subj['nonanimal'][i_subj] = []
        self.min_rt_subj['animal_wrong'][i_subj] = []
        self.min_rt_subj['nonanimal_wrong'][i_subj] = []
        # Organize trials per-image
        count = 0
        for t in trials:
            self.n_trials += 1
            stim = json.loads(t["stimulus"])
            # recover max_rt and im num in block
            max_rt = stim["duration"] - stim["onset"] - 50
            im_in_block = count % (len(trials) / 6)
            t['max_rt'] = max_rt
            t['im_in_block'] = im_in_block
            img = stim["stimulus"].split('/')[3]
            t['cat'] = stim["stimulus"].split('/')[3]
            # Get score 1/0 for true/wrong answer for accumulating accuracies
            if t['response'] == t['true_response']:
                t['score'] = 1
                if t['rt'] > 0:
                    self.min_rt_subj[self.im2lab[(t['cat'] + '.jpg')]][i_subj].append(t['rt'])
                    self.correct_rts.append(t['rt'])
            else:
                t['score'] = 0
                if t['rt'] > 0:
                    self.min_rt_subj[self.im2lab[(t['cat'] + '.jpg')] + '_wrong'][i_subj].append(t['rt'])

            self.im_by_subj[i_subj][img] = t['score']
            is_timeout = (t['rt'] < 0)
            if is_timeout: self.n_timeouts += 1
            if (not is_timeout) or (not self.ignore_timeouts):
                if is_timeout:
                    t['cscore'] = 0.5
                else:
                    t['cscore'] = t['score']
                if img in self.trials_by_img:
                    self.trials_by_img[img] += [t]
                else:
                    self.trials_by_img[img] = [t]
                if t['rt'] > 0:
                    self.total_acc.append(t['score'])
                    self.total_time.append(t['rt'])
                # store data by max_rt
                self.data_by_max_rt[str(max_rt)][im_in_block].append(t['rt'])
                self.responses_by_max_rt[str(max_rt)][im_in_block].append(t['score'])
                # Store per image data
                if img not in self.acc_by_im.keys():
                    self.acc_by_im[img] = {}
                    self.avg_acc_by_im[img] = [None, None, None]
                if str(max_rt) not in self.acc_by_im[img].keys():
                    self.acc_by_im[img][str(max_rt)] = []
                self.acc_by_im[img][str(max_rt)].append(t['score'])
            count += 1

    def eval_participants(self):
        # store image names with index in dict to hold model comparisons
        self.animal_ind = 0
        self.nonanimal_ind = 0
        for im in self.im2lab.keys():
            if self.im2lab[im] == 'animal':
                self.im2key[im] = self.animal_ind
                self.animal_ind +=1
            else:
                self.im2key[im] = self.nonanimal_ind
                self.nonanimal_ind +=1
        # Prepare model comparison dictionaries
        self.model_comp['animal'] = {}
        self.model_comp['nonanimal'] = {}
        self.model_comp['animal']['source_images'] = np.empty([self.animal_ind,1],dtype='S20')
        self.model_comp['nonanimal']['source_images'] = np.empty([self.nonanimal_ind,1],dtype='S20')
        for im in self.im2key.keys():
            self.model_comp[self.im2lab[im]]['source_images'][self.im2key[im]][0] = im
        self.model_comp['animal']['human_acc'] = np.empty([1,self.animal_ind]) #arrays to hold human accuracy data for model comparison
        self.model_comp['nonanimal']['human_acc'] = np.empty([1,self.nonanimal_ind])
        # caluclate mean RT and performance per image
        for img,trials in self.trials_by_img.iteritems():
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
            self.data_by_img[img] = data
            # Store data for model comparison
            if self.im2lab[img+'.jpg'] == 'animal':
                self.model_comp['animal']['human_acc'][0][self.im2key[img+'.jpg']] = float(score*100)
            else:
                self.model_comp['nonanimal']['human_acc'][0][self.im2key[img+'.jpg']] = float(score*100)

        self.score_data = [self.data_by_img[img]['score']*100 for img in self.data_by_img]
        self.rt_data = [self.data_by_img[img]['rt'] for img in self.data_by_img]

        #Analyze results
        # Track failure rates
        # rt hists by max_rt
        mrts = [(self.mrt_500, self.mrt_500_class, self.mrt_500_25, self.mrt_500_25_class),
                (self.mrt_1000, self.mrt_1000_class, self.mrt_1000_25, self.mrt_1000_25_class),
                (self.mrt_2000, self.mrt_2000_class, self.mrt_2000_25, self.mrt_2000_25_class)]
        for i_rt, rt in enumerate(self.rts):
            mrt, mrt_class, mrt_25, mrt_25_class = mrts[i_rt]
            for im_num in range(len(self.data_by_max_rt[rt])):
                for trial in range(len(self.data_by_max_rt[rt][0])):
                    self.failures[i_rt][1] += 1
                    if self.data_by_max_rt[rt][im_num][trial] > 0:
                        mrt.append(self.data_by_max_rt[rt][im_num][trial])
                        mrt_class.append(self.responses_by_max_rt[rt][im_num][trial])
                        if im_num >23:
                            mrt_25.append(self.data_by_max_rt[rt][im_num][trial])
                            mrt_25_class.append(self.responses_by_max_rt[rt][im_num][trial])
                    else:
                        self.failures[i_rt][0] +=1

        # Average rt over time
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Not all RTs occur in experiments
            for im_num in range(len(self.responses_by_max_rt['500'])):
                temp_500 = []
                temp_1000 = []
                temp_2000 = []
                temp_500_class = []
                temp_1000_class = []
                temp_2000_class = []
                for trial in range(len(self.responses_by_max_rt['500'][0])):
                    if self.data_by_max_rt['500'][im_num][trial] > 0:
                        temp_500.append(self.data_by_max_rt['500'][im_num][trial])
                        temp_500_class.append(self.responses_by_max_rt['500'][im_num][trial])
                for trial in range(len(self.responses_by_max_rt['1000'][0])):
                    if self.data_by_max_rt['1000'][im_num][trial] > 0:
                        temp_1000.append(self.data_by_max_rt['1000'][im_num][trial])
                        temp_1000_class.append(self.responses_by_max_rt['1000'][im_num][trial])
                for trial in range(len(self.responses_by_max_rt['2000'][0])):
                    if self.data_by_max_rt['2000'][im_num][trial] > 0:
                        temp_2000.append(self.data_by_max_rt['2000'][im_num][trial])
                        temp_2000_class.append(self.responses_by_max_rt['2000'][im_num][trial])
                self.mrt_500_avgrt.append(np.mean(temp_500))
                self.mrt_1000_avgrt.append(np.mean(temp_1000))
                self.mrt_2000_avgrt.append(np.mean(temp_2000))
                self.mrt_500_avgclass.append(np.mean(temp_500_class)*100)
                self.mrt_1000_avgclass.append(np.mean(temp_1000_class)*100)
                self.mrt_2000_avgclass.append(np.mean(temp_2000_class)*100)
            # Average accuracy per image by max_rt
            for im in self.acc_by_im.keys():
                for i_rt, rt in enumerate(self.rts):
                    if rt in self.acc_by_im[im]:
                        self.avg_acc_by_im[im][i_rt] = np.mean(self.acc_by_im[im][rt])*100

        # Find failure rates
        self.fail_500 = float(self.failures[0][0])*100/max(1.0, float(self.failures[0][1]))
        self.fail_1000 = float(self.failures[1][0])*100/max(1.0, float(self.failures[1][1]))
        self.fail_2000 = float(self.failures[2][0])*100/max(1.0, float(self.failures[2][1]))

        # find human accuracy
        self.hum_im_acc = []
        for im in sorted(self.acc_by_im.keys()):
            res = []
            for mrt in self.acc_by_im[im].keys():
                res = np.concatenate((res, self.acc_by_im[im][mrt]))
            self.hum_im_acc.append(np.mean(res))

    def load_model_data(self, model, classifier_type, train_batches):
        layer_names = util.get_model_layers(model)
        for layer in layer_names:
            self.load_model_layer_data(model, layer, classifier_type, train_batches)

    def load_model_layer_data(self, model, layer, classifier_type, train_batches):
        accs = []
        self.model_comp['animal'][model + '_' + layer] = np.ones([1, self.animal_ind])
        self.model_comp['nonanimal'][model + '_' + layer] = np.ones([1, self.nonanimal_ind])
        n_images_for_acc = 0
        for set_index, set_name in self.loaded_sets:
            inputfn = util.get_predictions_filename(model, layer, classifier_type, train_batches, set_index, set_name)
            modeldata = np.load(inputfn)
            for index in range(len(modeldata['source_filenames'])):
                impath = modeldata['source_filenames'][index]
                imname = impath.split('/')[-1] + '.jpg'
                # if image was tested
                if imname in self.im2key.keys():
                    self.model_comp[self.im2lab[imname]][model + '_' + layer][0][self.im2key[imname]] = modeldata['hyper_dist'][index]
            acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels'])) / float(len(modeldata['pred_labels']))
            n_images_for_acc += len(modeldata['pred_labels'])
            accs += [acc]
        mean_acc = np.mean(acc)
        self.model_accs[model + '_' + layer] = mean_acc
        print '%.1f accuracy for %s_%s derived from %d images (%d sets: %s)' % (mean_acc*100, model, layer, n_images_for_acc, len(accs), str(self.loaded_sets))

    def calc_model_correlation(self, model, corr, adjust_for_true_label=True, verbose=True):
        layer_names = util.get_model_layers(model)
        for layer in layer_names:
            self.calc_model_layer_correlation(model, layer, corr, adjust_for_true_label=adjust_for_true_label, verbose=verbose)
        self.corr_type = corr

    def calc_model_layer_correlation(self, model, layer, corr, adjust_for_true_label=True, verbose=True):
        # Calculate correlation and significance
        if adjust_for_true_label:
            human_res = np.concatenate((self.model_comp['animal']['human_acc'][0], self.model_comp['nonanimal']['human_acc'][0]))
            model_res = np.concatenate((self.model_comp['animal'][model + '_' + layer][0], -1 * self.model_comp['nonanimal'][model + '_' + layer][0]))
        else:
            human_res = np.concatenate((self.model_comp['animal']['human_acc'][0], 50 - self.model_comp['nonanimal']['human_acc'][0]))
            model_res = np.concatenate((self.model_comp['animal'][model + '_' + layer][0], self.model_comp['nonanimal'][model + '_' + layer][0]))
        if corr == "Spearman's rho":
            corrl, p_val = stats.spearmanr(human_res, model_res)
        elif corr == "Pearson's r":
            corrl, p_val = stats.pearsonr(human_res, model_res)
        else:
            corrl, p_val = stats.kendalltau(human_res, model_res)
        if verbose: print "Correctness Corelation: %s, p-value: %s" % (str(corrl), str(p_val))
        self.model_corrs[model + '_' + layer] = (corrl, p_val)

    def eval_by_classes(self, class_idcs, class_names):
        self.model_comp_by_class = {}
        for class_idx,class_name in zip(class_idcs, class_names):
            imageset_filename = util.get_imageset_filename(class_idx, 'set')
            class_images = label_results(imageset_filename)
            comp = {'animal': [], 'nonanimal': []}
            rts = {'animal': [], 'nonanimal': []}
            for im,lbl in class_images.iteritems():
                if im in self.im2key:
                    comp[lbl] += [self.model_comp['animal']['human_acc'][0, self.im2key[im]]]
                    rts[lbl] += [self.data_by_img[im[:-4]]['rt']]
            self.model_comp_by_class[class_idx] = comp
            print 'Class %s: Found %d animal, %d nonanimal trials' % (class_name, len(comp['animal']), len(comp['nonanimal']))
            print 'Class %s percent correct: %.1f (animal=%.1f, nonanimal=%.1f)' % (class_name, np.mean(comp['animal']+comp['nonanimal']), np.mean(comp['animal']), np.mean(comp['nonanimal']))
            print 'Class %s RT: %.1fms (animal=%.1fms, nonanimal=%.1fms)' % (class_name, np.mean(rts['animal'] + rts['nonanimal']), np.mean(rts['animal']),np.mean(rts['nonanimal']))
