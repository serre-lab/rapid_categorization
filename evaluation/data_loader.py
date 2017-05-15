#!/usr/bin/env python
# Load experimental and model data

import re
import sqlite3
import json
import numpy as np
from rapid_categorization.model import util
from results_key import label_results
from scipy import stats
from collections import defaultdict
from rapid_categorization.clicktionary.config import experiment_descs
from rapid_categorization.run_settings.settings import get_settings

class Data:
    def __init__(self):
        self.im2lab = {}
        self.model_comp = {} # data for model comparison
        self.im2key = {}  # maps image to index in model comp array
        self.im2path = {}  # maps image filename to containing folder
        self.animal_ind = 0
        self.nonanimal_ind = 0
        self.timeout_threshold = 20
        self.trials_by_img = {}  # trial data indexed by image
        self.data_by_img = {}  # result data averaged over trials by image
        self.acc_by_im = {}
        self.acc_by_im_subjects = {}
        self.acc_by_im_and_max_answer_time = {} # Dict indexed by image filename; contains dictionaries indexed by max answer time
        self.per_subject_rev_acc = {}        
        self.total_acc = []
        self.total_time = []
        self.min_rt_subj = {}
        self.min_rt_subj['animal'] = {}
        self.min_rt_subj['nonanimal'] = {}
        self.min_rt_subj['animal_wrong'] = {}
        self.min_rt_subj['nonanimal_wrong'] = {}
        self.im_by_subj = {}
        self.sub_timeouts = {}
        self.max_answer_times = set()
        # Track failure rates
        # rt hists by max_rt
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
        self.workerIds = []
        self.statuses = (3, 4, 5, 7)  # See https://github.com/NYUCCL/psiTurk/blob/master/psiturk/experiment.py
        self.full_accuracies = {}
        self.response_log = {}
        self.force_remove_duplicate_workers = False  # True
        self.worker_list = []

    def load(self, experiment_run, exclude_workerids=['']):
        p = get_settings(experiment_run)
        set_index, set_name = p['set_index'], p['set_name']
        self.load_ground_truth(set_index, set_name)
        self.load_participants(experiment_run, exclude_workerids)

    def load_ground_truth(self, set_index, set_name):
        imageset_filename = util.get_imageset_filename(set_index, set_name)
        # Generate ground truth key
        self.im2lab.update(label_results(imageset_filename))
        self.loaded_sets += [(set_index, set_name)]

    def load_participant_json(self, experiment_run, verbose=True):
        exp_filename = util.get_experiment_db_filename_by_run(experiment_run)
        r = sqlite3.connect(exp_filename).cursor().execute(
            "SELECT workerid,beginhit,status,datastring FROM placecat WHERE status in %s AND NOT datastring==''" % (self.statuses,)).fetchall()
        if verbose: print "%d participants found in file %s." % (len(r), exp_filename)
        return r

    def load_participants(self, experiment_run, exclude_workers):
        r = self.load_participant_json(experiment_run, verbose=True)
        for i_subj in range(0,len(r)):
            data = json.loads(r[i_subj][3])
            sub_id = json.loads(r[i_subj][3])['workerId']
            if sub_id not in exclude_workers:
                self.load_subject(data)
            else:
                print 'Excluding subject: %s' % sub_id

    def get_participant_ids(self, experiment_run):
        r = self.load_participant_json(experiment_run, verbose=True)
        print experiment_run
        for i_subj in range(0,len(r)):
            self.workerIds.append(json.loads(r[i_subj][3])['workerId'])

    def load_im2path(self, experiment_ids):
        self.im2path = {}
        for experiment_id in experiment_ids:
            set_index, set_name = util.get_experiment_imageset(experiment_id)
            imageset_filename = util.get_imageset_filename(set_index, set_name)
            im_names = label_results(imageset_filename)
            im_root = util.get_input_image_root(set_index, set_name)
            for k in im_names.iterkeys():
                self.im2path[k] = im_root

    def load_multi(self, set_indexes, set_name, bootstrap_size=None):
        r = []
        for set_index in set_indexes:
            self.load_ground_truth(set_index, set_name)
            r += self.load_participant_json(set_index, set_name, verbose=(bootstrap_size is None))
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
        self.im_by_subj[i_subj] = {}
        self.per_subject_rev_acc[str(i_subj)] = {}
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
        if self.force_remove_duplicate_workers and data['workerId'] in self.worker_list:
            print 'Removing duplicate instance of worker %s' % data['workerId']
        else:
            self.worker_list.append(data['workerId'])
            t_array = []
            # Organize trials per-image
            count = 0
            sub_timeout = 0
            try:
                for t in trials:
                    self.n_trials += 1
                    stim = json.loads(t["stimulus"])
                    if 'examples' in stim["stimulus"]: # Example video not used in evaluation
                        continue
                    max_answer_time = stim['duration'] - stim['onset']
                    t['max_answer_time'] = max_answer_time
                    self.max_answer_times.add(max_answer_time)
                    # recover max_rt and im num in block
                    max_rt = stim["duration"] - stim["onset"] - 50
                    im_in_block = count % (len(trials) / 6)
                    t['max_rt'] = max_rt
                    t['im_in_block'] = im_in_block
                    img = '/'.join(stim["stimulus"].split('/')[3:-1])
                    t['cat'] = img
                    # Get score 1/0 for true/wrong answer for accumulating accuracies
                    if t['response'] == t['true_response']:
                        t['score'] = 1.0
                        if t['rt'] > 0:
                            self.min_rt_subj[self.im2lab[img]][i_subj].append(t['rt'])
                            self.correct_rts.append(t['rt'])
                    else:
                        t['score'] = 0.0
                        if t['rt'] > 0:
                            self.min_rt_subj[self.im2lab[img] + '_wrong'][i_subj].append(t['rt'])

                    self.im_by_subj[i_subj][img] = t['score']
                    is_timeout = (t['rt'] < self.timeout_threshold)
                    if is_timeout:
                        self.n_timeouts += 1
                        sub_timeout += 1
                    if (not is_timeout) or (not self.ignore_timeouts):
                        if is_timeout:
                            t['cscore'] = 0.5
                            print 'here'
                        else:
                            t['cscore'] = t['score']
                        if img in self.trials_by_img:
                            self.trials_by_img[img] += [t]
                        else:
                            self.trials_by_img[img] = [t]
                        if t['rt'] > 0:
                            self.total_acc.append(t['score'])
                            self.total_time.append(t['rt'])
                        # Store per image data
                        if img not in self.acc_by_im.keys():
                            self.acc_by_im[img] = []
                            self.acc_by_im_subjects[img] = []
                            self.acc_by_im_and_max_answer_time[img] = {}
                        t_array.append(t)
                        self.acc_by_im[img].append(t['score'])
                        self.acc_by_im_subjects[img].append(i_subj)
                        if max_answer_time not in self.acc_by_im_and_max_answer_time[img]:
                            self.acc_by_im_and_max_answer_time[img][max_answer_time] = []
                        self.acc_by_im_and_max_answer_time[img][max_answer_time].append(t['score'])
                        # self.get_image_names(t['cat']) index is rev/imagename/category
                        if str(self.get_image_names(t['cat'])[0]) in self.per_subject_rev_acc[str(i_subj)].keys():
                            self.per_subject_rev_acc[str(i_subj)][self.get_image_names(t['cat'])[0]] += [t['score']]
                        else:
                            self.per_subject_rev_acc[str(i_subj)][self.get_image_names(t['cat'])[0]] = [t['score']]

                    count += 1
                self.full_accuracies[i_subj] = (np.mean(
                    [r['score']
                        for r in t_array if r['cat'].split('/')[0] == 'full']))
                self.response_log[i_subj] = [r['key_press'] for r in t_array]
                self.n_subjects += 1
                self.sub_timeouts[i_subj] = sub_timeout
                assert len(self.full_accuracies.keys()) == self.n_subjects
            except:
                print 'Fucked up on trial: %s for participant" %s' % (
                    count, data['workerId'],)

    def get_image_names(self, im):
        rev, base_im = im.split('/')
        if rev ==  'full':
            rev = -10
        base_im_name = re.findall('[a-zA-Z_]*', base_im)[0]
        return rev, base_im, base_im_name

    def create_image_label(self, im, num_subjects):
        vals = 0
        for x, modifier in zip(re.findall('\d+', im), 1 + np.arange(len(re.findall('\d+', im)))):
            vals += int(x) * modifier
        return list(np.repeat(vals, num_subjects))

    def get_summary_by_revelation(self, filename_filter=None, subject_filter=None):
        rev_scores = defaultdict(list)
        rev_subs = defaultdict(list)
        rev_ims = defaultdict(list)
        rev_ims = defaultdict(list)
        for (im, scores), (im_check, subjects) in zip(
                self.acc_by_im.iteritems(), self.acc_by_im_subjects.iteritems()):
            rev, base_im, base_im_name = self.get_image_names(im)
            _, _, base_im_name_check = self.get_image_names(im_check)
            assert base_im_name == base_im_name_check
            assert len(scores) == len(subjects)
            if filename_filter is not None:
                if base_im_name not in filename_filter:
                    continue
            if subject_filter is not None:
                filt_scores = [sc for sc, su in zip(
                    scores, subjects) if su in subject_filter]
                filt_subs = [su for su in subjects if su in subject_filter]
                scores = filt_scores
                subjects = filt_subs
            rev_scores[int(rev)] += scores
            rev_subs[int(rev)] += subjects
            rev_ims[int(rev)] += [im] * len(scores) # self.create_image_label(im, len(scores))

        self.per_subject_rev_acc_mean = {}
        for k, v in self.per_subject_rev_acc.iteritems():
            # Per subject\
            if subject_filter is not None:
                if int(k) in subject_filter:
                    self.per_subject_rev_acc_mean[k] = {}
                    for j, w in v.iteritems():
                        self.per_subject_rev_acc_mean[k][j] = np.mean(w)

        revs = sorted(rev_scores.keys())
        all_scores = [rev_scores[irev] for irev in revs]
        all_subjects = [rev_subs[irev] for irev in revs]
        all_ims = [rev_ims[irev] for irev in revs]
        return revs, all_scores, all_subjects, all_ims

    def get_summary_by_revelation_and_max_answer_time(self, max_answer_time, filename_filter=None):
        rev_scores = defaultdict(list)
        for im, scores_by_max_answer_time in self.acc_by_im_and_max_answer_time.iteritems():
            scores = scores_by_max_answer_time.get(max_answer_time, [])
            rev, base_im = im.split('/')
            if rev ==  'full':
                rev = -10
            base_im_name = re.findall('[a-zA-Z_]*', base_im)[0]
            if filename_filter is not None:
                if base_im_name not in filename_filter:
                    continue
            rev_scores[int(rev)] += scores
        revs = sorted(rev_scores.keys())
        all_scores = [rev_scores[rev] for rev in revs]
        return revs, all_scores

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
            if self.im2lab[img] == 'animal':
                self.model_comp['animal']['human_acc'][0][self.im2key[img]] = float(score*100)
            else:
                self.model_comp['nonanimal']['human_acc'][0][self.im2key[img]] = float(score*100)

        self.score_data = [self.data_by_img[img]['score']*100 for img in self.data_by_img]
        self.rt_data = [self.data_by_img[img]['rt'] for img in self.data_by_img]

        #Analyze results
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
        for set_index, set_name in self.loaded_sets:
            inputfn = util.get_predictions_filename(model, layer, classifier_type, train_batches, set_index, set_name)
            modeldata = np.load(inputfn)
            for index in range(len(modeldata['source_filenames'])):
                impath = modeldata['source_filenames'][index]
                imname = impath.split('/')[-1]
                # if image was tested
                if imname in self.im2key.keys():
                    self.model_comp[self.im2lab[imname]][model + '_' + layer][0][self.im2key[imname]] = modeldata['hyper_dist'][index]
            acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels'])) / float(len(modeldata['pred_labels']))
            accs += [acc]
        self.model_accs[model + '_' + layer] = np.mean(acc)

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


if __name__ == '__main__':
    """Test: Check for duplicate subs."""
    data = Data()
    data.ignore_timeouts = True
    # data.load_ground_truth(set_index=50, set_name='clicktionary')
    # revs, scores = data.get_summary_by_revalation()
    # print revs
    # print scores
    data.load('click_center_probfill_400stim_150res')
    for k in sorted([k.split('/')[1] for k in data.acc_by_im.keys()]):
        print k
