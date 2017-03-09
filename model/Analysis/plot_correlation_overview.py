#!/usr/bin/env python

from rapid_categorization.levels import util
import numpy as np
import matplotlib.pyplot as plt
from plot_correlations import plot_corr_correct

def overviewPlot(data, model_names, axis_type, use_bootstrap_value, plot_corr_errs, use_subplots, is_unadjusted):
    acc_title = 'Model Accuracy'
    if use_subplots:
        corr_title = 'Correlation with Humans'
    else:
        corr_title = 'Correlation with Humans'
    if is_unadjusted:
        save_prefix = 'unadjusted_'
        unadjusted_suffix = ' (Uncorrected)'
    else:
        save_prefix = ''
        unadjusted_suffix = ''
    if len(model_names) == 1:
        acc_title = util.get_model_human_name(model_names[0], True)+' ' + acc_title
        save_prefix += model_names[0]
    else:
        save_prefix += 'comp'
    if not isinstance(plot_corr_errs, list):
        plot_corr_errs = [plot_corr_errs] * len(model_names)
    if use_subplots:
        fig, axs = plt.subplots(1, 2)
        ax1 = axs[0]
        ax2 = axs[1]
        ax2.yaxis.tick_right()
        acc_tick_color = 'k'
        corr_tick_color = 'k'
        ax1.set_title(acc_title)
        ax2.set_title(corr_title)
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(bottom=0.1, top=0.8)
        fig.suptitle('Model Comparison' + unadjusted_suffix, fontsize=14, weight='bold')
        if not is_unadjusted:
            ax1.text(-0.05, 1.03, '(a)', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14, weight='bold')
            # Somehow cannot align this with ax2
            ax1.text(1.10, 1.03, '(b)', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14, weight='bold')
    else:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        axs = [ax1] # Only need to modify one x axis
        acc_tick_color = 'b'
        corr_tick_color = 'r'
        fig.set_size_inches(8 + is_unadjusted * 2, 6)
        fig.subplots_adjust(bottom=0.2, top=0.92)
        plt.title(acc_title + ' and ' + corr_title + unadjusted_suffix, fontsize=14, weight='bold', y=1.02)
        if not is_unadjusted:
            ax1.text(-0.07, 1.04, '(c)', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=14, weight='bold')
    n_layers_max = 0
    x_max = 0
    x_indent = 0.5
    for model_name,plot_corr_err in zip(model_names, plot_corr_errs):
        # Get layer info
        layer_names = util.get_model_layers(model_name)
        layer_names_short = util.get_model_layers(model_name, True)
        n_layers = len(layer_names)
        n_layers_max = max(n_layers_max, n_layers)
        # Collect correlation and accuracy
        corrs = np.ones((1,n_layers))
        if plot_corr_errs: corr_errs = np.ones((2,n_layers))
        accs = np.ones((1,n_layers))
        for ind,layer_name in enumerate(layer_names):
            lidx = model_name + '_' + layer_name
            if use_bootstrap_value and lidx in data.model_corrs_bootstrapped:
                corrs_all = data.model_corrs_bootstrapped[lidx]
                corr_errs[0][ind] = np.percentile(corrs_all, 2.5)
                corr_errs[1][ind] = np.percentile(corrs_all, 97.5)
                corrs[0][ind] = np.median(corrs_all)
            else:
                if use_bootstrap_value:
                    print 'Warning: %s not bootstrapped!' % lidx
                corrs[0][ind] = data.model_corrs[lidx][0]
            accs[0][ind] = data.model_accs[lidx]
        # Define x axis
        if axis_type == 'rf':
            x = util.get_model_rf_sizes(model_name)
        elif axis_type == 'ridx':
            x = np.linspace(0, 1, n_layers)
            x_indent /= 5
        else:
            x = xrange(n_layers)
        x_max = max(x_max, x[-1])
        # Plot accuracy
        marker = util.get_model_plot_marker(model_name)
        acc_color = util.get_model_plot_color(model_name, plot_type='acc' if not use_subplots else None)
        ax1.plot(x,accs[0][:]*100,marker,color=acc_color)
        ax1.plot([], [], '-'+marker, color=acc_color, label=util.get_model_human_name(model_name))
        p1_fit = np.polyfit(x,accs[0][:]*100,2)
        p1_fn = np.poly1d(p1_fit)
        xs = np.linspace(x[0], x[-1])
        ax1.plot(xs,p1_fn(xs),color=acc_color)
        # Plot correlation
        corr_color = util.get_model_plot_color(model_name, plot_type='corr' if not use_subplots else None)
        if plot_corr_err:
            corr_err_color = util.get_model_plot_color(model_name, plot_type='corr' if not use_subplots else None, brighten=True)
            fits = [None]*2
            for ierr in (0,1):
                p2_fit = np.polyfit(x, corr_errs[ierr][:], 3)
                p2_fn = np.poly1d(p2_fit)
                fits[ierr] = p2_fn(xs)
            ax2.fill_between(xs, fits[0], fits[1], where=fits[1] >= fits[0], color='none', facecolor=corr_err_color, interpolate=True)
        ax2.plot(x,corrs[0][:],marker,color=corr_color)
        p2_fit = np.polyfit(x,corrs[0][:],3)
        p2_fn = np.poly1d(p2_fit)
        ax2.plot(xs,p2_fn(xs),color=corr_color)
        peak_corr_index = np.argmax(corrs[0])
        corr_range = (corr_errs[1][peak_corr_index] - corr_errs[0][peak_corr_index])/2
        print '%s peak correlation: %.3f pm %.3f' % (model_name, corrs[0][peak_corr_index], corr_range)
    # Plot human accuracy
    if use_bootstrap_value:
        hum_acc_errs = np.percentile(data.human_acc_bootstrapped, [2.5, 97.5])
        hum_acc = np.median(data.human_acc_bootstrapped) * 100
        for hum_acc_err in hum_acc_errs:
            ax1.plot([-x_indent, x_max + x_indent], [hum_acc_err*100, hum_acc_err*100], color='0.5', ls='--')
    else:
        hum_acc = int(np.mean(data.hum_im_acc) * 100)
    ax1.plot([-x_indent, x_max + x_indent], [hum_acc, hum_acc], color='0.5', ls='-')
    # Accuracy axis
    ax1.set_ylim([60 if use_subplots else 48, 100])
    ax1.set_ylabel('Accuracy (%)', color=acc_tick_color)
    if use_subplots:
        ax1.text(x_max * 1.9 / 4, hum_acc - 3, 'Human Accuracy', color='.5')
    else:
        ax1.text(x_max*3/4, hum_acc+3, 'Human Accuracy', color='.5')
    for tick in ax1.get_yticklabels():
        tick.set_color(acc_tick_color)
    # Correlation axis
    ax2.set_ylim([0, 0.45 if (use_subplots and not is_unadjusted) else 1])
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(data.corr_type + ' Correlation', color=corr_tick_color)
    if not use_subplots:
        for tick in ax2.get_yticklabels():
            tick.set_color(corr_tick_color)
    # x axis
    for ax in axs:
        ax.set_xlim([-x_indent, x_max + x_indent])
        if axis_type == 'rf':
            ax.set_xlabel('Receptive field size')
        elif axis_type == 'idx':
            ax.set_xlabel('Layer index')
        elif axis_type == 'ridx':
            ax.set_xlabel('Relative layer depth')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names_short, rotation=70)
            ax.set_xlabel('Layer (Increasing Complexity)')

    # Legend
    if use_subplots:
        ax1.legend(bbox_to_anchor=(0., 1.07, 2.2, 0.102), ncol=4, mode='expand', borderaxespad=0)

    # Done
    fn = util.at_plot_path(save_prefix + '_' + axis_type + '_overview.pdf')
    plt.savefig(fn)
    plt.savefig(util.at_plot_path(save_prefix + '_' + axis_type + '_overview.png'))
    print 'Saved to %s' % fn

