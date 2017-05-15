import os
import re
from glob import glob
from scipy import misc
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def MIRC_point(means, chance=0.5):
    mean_diffs = means.correctness.diff()
    mean_diffs[means.correctness < chance] = np.nan
    MIRC_point = mean_diffs.idxmax()
    sub_MIRC_point = MIRC_point - 1
    return sub_MIRC_point, MIRC_point


def plot_MIRC_sub_MIRC(MIRC_im, sub_im, score, output_dir):
    fig = plt.figure()
    label = MIRC_im.split('/')[-1]
    plt.subplot(1, 2, 1)
    MIRC = np.repeat(misc.imread(MIRC_im)[:, :, None], 3, axis=-1)
    plt.imshow(MIRC)
    plt.title('%s MIRC: %s%% accuracy' % (
        label, np.around(score['MIRCs'] * 100, 2)))
    plt.subplot(1, 2, 2)
    sub_MIRC = np.repeat(misc.imread(sub_im)[:, :, None], 3, axis=-1)
    plt.imshow(sub_MIRC)
    plt.title('%s sub-MIRC: %s%% accuracy' % (
        label, np.around(score['sub_MIRCs'] * 100, 2)))
    output_file = os.path.join(
        output_dir, 'mirc_panel_' + MIRC_im.split('/')[-1])
    plt.savefig(output_file)
    np.savez(os.path.join(
        output_dir, 'mirc_panel_' + MIRC_im.split('/')[-1]),
        sub_MIRC=sub_MIRC,
        MIRC=MIRC,
        **score)
    plt.close(fig)
    return {
            '%s/%s/%s' % (
                score['MIRCs'],
                score['sub_MIRCs'],
                label): [
                sub_MIRC, MIRC],
            }


def generate_MIRC_panels(
        ims,
        sub_MIRCs,
        MIRCs,
        score,
        image_dir,
        output_dir,
        keep_categories,
        im_ext='.png'):
    dirs = sorted(glob(os.path.join(image_dir, '*')))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    names = [x.split('/')[-1] for x in ims]
    MIRC_images = list(set(
        [x for x in glob(os.path.join(dirs[MIRCs], '*' + im_ext))
            for na in names if na in x]))
    sub_MIRC_images = [os.path.join(
        dirs[sub_MIRCs], x.split('/')[-1]) for x in MIRC_images]
    MIRC_dict = {}
    for m, s in tqdm(
            zip(MIRC_images, sub_MIRC_images), total=len(MIRC_images)):
        it_MIRCs = plot_MIRC_sub_MIRC(m, s, score, output_dir)
        if keep_categories is not None:
            split_string = re.split('\d', it_MIRCs.keys()[0].split('/')[-1])
            if split_string[0] in keep_categories:
                MIRC_dict.update(it_MIRCs)
                # Only take the first image
                keep_categories.pop(keep_categories.index(split_string[0]))
        else:
            MIRC_dict.update(it_MIRCs)
    return MIRC_dict


def clear_labels(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def produce_MIRC_figure(MIRC_dict, output):
    fig = plt.figure(figsize=(16, 8))
    num_images = len(MIRC_dict.keys())
    gs = gridspec.GridSpec(2, num_images, wspace=0.1, hspace=0.0)
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 10
    for idx, (g, (k, v)) in enumerate(zip(gs, MIRC_dict.iteritems())):
        ax = plt.subplot(g)
        ax.set_title('%s\n' % re.split(
            '\d', k.split('/')[-1])[0].replace('_', ' ').capitalize() + \
            r'$\Delta$ %.2f%%' % (
                float(k.split('/')[0]) * 100 - float(k.split('/')[1]) * 100))
        ax.imshow(v[0])
        clear_labels(ax)
        ax1 = plt.subplot(gs[idx + num_images])
        ax1.imshow(v[1])
        clear_labels(ax1)
        if idx == 0:
            ax.set_ylabel('sub-MIRC')
            ax1.set_ylabel('MIRC')
    plt.savefig(output)
    plt.close(fig)


def find_MIRCS(
        data,
        ims,
        image_dir,
        output_dir,
        generate_per_rev=True,
        generate_per_category=True,
        keep_categories=[
            'great_white_shark',
            'sorrel',
            'speedboat',
            'sportscar',
        ]
        ):
    df = pd.DataFrame(
        data,
        columns=['Revelation', 'correctness', 'subject'])
    image_categories = set(
        [re.split('\d+', x.split('/')[-1])[0] for x in ims[0]])
    df['images'] = np.concatenate(ims)
    if generate_per_rev:
        rev_means = df.groupby(['Revelation'], sort=True).aggregate(
            np.mean).add_suffix('').reset_index()
        group_sub_MIRCs, group_MIRCs = MIRC_point(rev_means)
        print 'Generating MIRC panels by across-revelation performance.'
        scores = {
            'MIRCs': rev_means.iloc[group_MIRCs].correctness,
            'sub_MIRCs': rev_means.iloc[group_sub_MIRCs].correctness,
            'MIRC_bin': group_MIRCs
        }
        images = generate_MIRC_panels(
            ims=ims[0],
            sub_MIRCs=group_sub_MIRCs,
            MIRCs=group_MIRCs,
            score=scores,
            image_dir=image_dir,
            output_dir=os.path.join(output_dir, 'rev_bins'),
            keep_categories=deepcopy(keep_categories))
        produce_MIRC_figure(images, os.path.join(
            output_dir, 'per_revelation.png'))

    if generate_per_category:
        image_means = df.groupby(
            ['Revelation', 'images'], sort=True).aggregate(
            np.mean).add_suffix('').reset_index()
        images = {}
        for cat in image_categories:
            print 'Working on %s' % cat
            im_means = image_means[image_means.images.str.contains(
                cat)].groupby(
                ['Revelation'], sort=True).aggregate(
                np.mean).add_suffix('').reset_index()
            group_sub_MIRCs, group_MIRCs = MIRC_point(im_means)
            scores = {
                'MIRCs': im_means.iloc[group_MIRCs].correctness,
                'sub_MIRCs': im_means.iloc[group_sub_MIRCs].correctness
            }
            filtered_ims = [i for i in ims[0] if cat in i]
            images.update(
                generate_MIRC_panels(
                    ims=filtered_ims,
                    sub_MIRCs=group_sub_MIRCs,
                    MIRCs=group_MIRCs,
                    score=scores,
                    image_dir=image_dir,
                    output_dir=os.path.join(output_dir, cat),
                    keep_categories=deepcopy(keep_categories)
                    )
            )

        produce_MIRC_figure(images, os.path.join(
            output_dir, 'per_category.png'))
