#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import numpy as np
from rt_plot import gen_RT_data, color_data
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

markers_data = ['o', 's', 'X', 'P', 'v', 'D']
def prepScatterData(all_fxn_data, all_fxn_seq):
    x = []
    y = []
    legends = []
    markers = []
    marker_color = []
    task_done = []
    exp_cat = []

    for n in range(len(all_fxn_seq)):
        for j in all_fxn_seq[n]:
            if all_fxn_data[j].task in task_done:
                pass
            else:
                x.append(all_fxn_data[j].hum_slope)
                y.append(all_fxn_data[j].model_slope)
                task_done.append(all_fxn_data[j].task)
                legends.append(all_fxn_data[j].task)
                markers.append(markers_data[n])
                marker_color.append(colors[color_data[all_fxn_data[j].task]])
                exp_cat.append(n)

    return x, y, markers, legends, marker_color, exp_cat


def createScatterPlots(method, dir, rt_fxn_model, task_ids, legends=False):
    plt.rcParams.update({'font.size': 11})
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    plt.figure(figsize=(4.125/1.3, 4.125/1.3))

    ax = plt.subplot(111)

    all_fxn_data, all_fxn_seq = gen_RT_data(method, dir, rt_fxn_model, task_ids)
    x, y, markers, legends, marker_color, exp_cat = prepScatterData(all_fxn_data, all_fxn_seq)

    maxv = max(x+y)
    minv = min(x+y)
    plt.plot([*range(int(minv), int(maxv+10), 10)], [*range(int(minv), int(maxv+10), 10)], '--', c='#BDBDBD', lw=1)

    last_exp_cat = -1
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=marker_color[i], label=legends[i], marker=markers[i])

        if last_exp_cat == exp_cat[i]:
            plt.plot([x[i-1], x[i]], [y[i-1], y[i]], '--', c='#818181', lw=0.3/2)

        last_exp_cat = exp_cat[i]

    plt.axis('square')
    plt.axis('equal')

    locs = np.arange(0, 101, 25)
    plt.yticks(locs)
    plt.xticks(locs)

    plt.xlabel("Human slope (ms / object)")
    plt.ylabel("Model slope (ms / object)")

    if legends:
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    try:
        os.makedirs("results/slope_scatter_plots")
    except:
        pass

    plt.savefig("results/slope_scatter_plots/" + method + ".pdf", dpi=200, bbox_inches="tight")
    plt.close()
