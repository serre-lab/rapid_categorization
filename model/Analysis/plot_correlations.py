#!/usr/bin/env python

from rapid_categorization.levels import util
import numpy as np
import matplotlib.pyplot as plt
import decimal


class Error(Exception):
   """Base class for other exceptions"""
   pass

class NotEveryImageSeenAtThisRTError(Error):
   """Raised when an image from the set wasn't seen at a given response time"""
   pass

def plot_corr_correct(data, model, layer):
    plt.figure()
    ax = plt.gca()
    plt.xlim((-10,110))
    layer_names = util.get_model_layers(model)
    layer_names_short = util.get_model_layers(model, True)
    ind = layer_names.index(layer)
    layer1 = layer_names_short[ind]
    if layer == 'fc7ex': plt.ylim((-2,4))
    elif layer == 'conv5_2ex': plt.ylim((-2,4)) #VGG16 conv5_2
    elif layer == 'conv5_1ex': plt.ylim((-0.1,1.1))
    elif layer == 'conv4_3ex': plt.ylim((-0.1,1.1))
    else: plt.ylim((-2,4))
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    plt.plot([xmin,xmax],[0,0], c='k', hold=True)
    plt.plot([0,0],[ymin,ymax], c='k', hold=True)
    plt.scatter(data.model_comp['animal']['human_acc'][0],data.model_comp['animal'][model+'_'+layer][0],c='b',label='Animal',hold=True)
    plt.scatter(data.model_comp['nonanimal']['human_acc'][0],-1*data.model_comp['nonanimal'][model+'_'+layer][0],c='g',label='Non-Animal',hold=True)
    plt.plot([50,50],[ymin,ymax], c='r',ls='dashed', hold=True, label='Chance')
    #relabel axis
    x_min = 0
    x_max = 101
    plt.xticks(np.arange(x_min,x_max,50))
    xlabs,xlocs = plt.xticks()
    plt.xticks(np.arange(x_min,x_max,50),abs(xlabs).astype(int))
    plt.legend(loc="upper left")
    plt.title('Fitting Human to '+model+' '+layer1+' Accuracy')
    plt.xlabel('Human Accuracy (%)')
    plt.ylabel('Distance from Hyperplane')

    # Get correlation and significance
    corrl, p_val = data.model_corrs[model+'_'+layer]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    plt.figtext(0.5, 0.05, 'Correlation: %.3f (%s), p-value: %.2E' %(corrl,data.corr_type,decimal.Decimal(str(p_val))),horizontalalignment='center')

    # Save figure
    plt.savefig(util.at_plot_path('corr_%s_%s.pdf' % (model, layer)))
    plt.savefig(util.at_plot_path('corr_%s_%s.png' % (model, layer)))

def plot_corr_correct_mrt(data, model_name, layer, corr, graphsOn=True):
    modeldata = np.load(inputfn)
    model_comp['animal'][model+'_'+layer] = np.ones([1,animal_ind])
    model_comp['nonanimal'][model+'_'+layer] = np.ones([1,nonanimal_ind])
    for index in range(len(modeldata['source_filenames'])):
        impath = modeldata['source_filenames'][index]
        imname = impath.split('/')[-1]+'.jpg'
        # if image was tested
        if imname in im2key.keys():
            model_comp[im2lab[imname]][model+'_'+layer][0][im2key[imname]] = modeldata['hyper_dist'][index]
    if graphsOn:
        ax = plt.gca()
        plt.xlim((-10,110))
    if model == 'VGG16':
        mname = model
        if layer == 'fc7ex':
#            plt.ylim((-5,5)) #VGG16 fc7
            plt.ylim((-2,4))
            ind = 14
        elif layer == 'fc6ex':
#            plt.ylim((-10,15)) #VGG16 fc6
            ind = 13
        elif layer == 'conv5_3ex':
#            plt.ylim((-60,50)) #VGG16 conv5_3
            ind = 12
        elif layer == 'conv5_2ex':
            plt.ylim((-1,3.5)) #VGG16 conv5_2
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
#            plt.ylim((-250, 500)) #VGG16 conv4_2
            ind = 8
        elif layer == 'conv4_1ex':
#            plt.ylim((-450,600)) #VGG16 conv4_1
            ind = 7
        elif layer == 'conv3_3ex':
#            plt.ylim((-450,800)) #VGG16 conv3_3
            ind = 6
        elif layer == 'conv3_2ex':
#            plt.ylim((-800,1000)) #VGG16 conv3_2
            ind = 5
        elif layer == 'conv3_1ex':
#            plt.ylim((-650,1000)) #VGG16 conv3_1
            ind = 4
        elif layer == 'conv2_2ex':
#            plt.ylim((-650,800)) #VGG16 conv2_2
            ind = 3
        elif layer == 'conv2_1ex':
#            plt.ylim((-600,600)) #VGG16 conv2_1
            ind = 2
        elif layer == 'conv1_2ex':
#            plt.ylim((-450,500)) #VGG16 conv1_2
            ind = 1
        else:
#            plt.ylim((-80,80)) #VGG16 conv1_1
            plt.ylim((-2,4))
            ind =0
        max_rts = ['500','1000','2000']
        for mrt in max_rts:
            model_comp['animal']['human_acc_'+mrt] = np.empty([1,nonanimal_ind])*-500
            model_comp['nonanimal']['human_acc_'+mrt] = np.empty([1,nonanimal_ind])*-500
            for index in range(len(modeldata['source_filenames'])):
                impath = modeldata['source_filenames'][index]
                imname = impath.split('/')[-1]+'.jpg'
                # if image was tested
                if imname in im2key.keys():
                    model_comp[im2lab[imname]]['human_acc_'+mrt][0][im2key[imname]] = float(np.mean(acc_by_im[imname.split('.')[0]][mrt]))*100

        row =1
        col = 3
        bad_sets = []
        for rt in max_rts:
            if len(filter(lambda a: a != -500, model_comp['animal']['human_acc_'+rt][0][:])) != len(model_comp['animal']['human_acc_'+rt][0][:]):
                bad_sets.append(rt+' animal')
            if len(filter(lambda a: a != -500, model_comp['nonanimal']['human_acc_'+rt][0][:])) != len(model_comp['animal']['human_acc_'+rt][0][:]):
                bad_sets.append(rt+' nonanimal')
        if len(bad_sets) >0:
            print "Not every image was seen in ", bad_sets
            raise NotEveryImageSeenAtThisRTError
        if graphsOn:
            f, subp =  plt.subplots(row, col, sharey='row',squeeze=False)
        count = 0
        corrs = [0 for i in range(len(max_rts))]
        for r in range(row):
            for c in range(col):
                if graphsOn:
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
                if graphsOn:
                    box = subp[r][c].get_position()
                    subp[r][c].set_position([box.x0, box.y0 + box.height * 0.15,box.width, box.height * 0.85])
                    f.text(box.x0+box.width/2,box.y0, 'Correlation: %.3f (%s), p-value: %.2E' %(corrl,corr,decimal.Decimal(str(p_val))), horizontalalignment='center')
                corrs[c] = corrl
                count +=1

                layer1 = ' '+layer[0:len(layer)-2]
        if graphsOn:
            f.suptitle('Fitting Human to ' +mname+' '+layer1+' Accuracy for Varied Max Response Times')
        acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels']))/float(len(modeldata['pred_labels']))
        print 'Accuracy: %.2f' %(acc)

        return (ind,corrs,acc)
