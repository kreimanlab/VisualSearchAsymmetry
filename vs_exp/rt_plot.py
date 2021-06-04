#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
import numpy as np
import os
from scipy import stats
import copy
from human_rt_data import rt_hum as human_rt_data
from mpl_toolkits.axisartist.axislines import Subplot
import matplotlib.pyplot as plt

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
blk_col = '#212121'
mpl.rc('axes', edgecolor=blk_col, labelcolor=blk_col)
mpl.rc('xtick', color=blk_col)
mpl.rc('ytick', color=blk_col)
mpl.rc('text', color=blk_col)

tasks = ['curvature', 'lightning_dir', 'intersection', 'intersection', 'categorization', 'categorization']
exp_names = ['Curvature (Exp 1)', 'Lighting Direction (Exp 2)', 'Intersection (cross vs non-cross, Exp 3)', 'Intersection (L vs T, Exp 4)', 'Orientation (homo, Exp 5)', 'Orientation (hetero, Exp 6)']

file_names = [['curve_in_lines', 'line_in_curves'],
              ['top_down', 'left_right'],
              ['non_cross', 'cross'],
              ['Ts', 'Ls'],
              ['homo_t20', 'homo_t0'],
              ['hetero_t0', 'hetero_t20']]

exp_tasks = [['Curve', 'Line'],
             ['Vertical', 'Horizontal'],
             ['No Intersection',  'Cross'],
             ['T', 'L'],
             ['T: 20 (homo)', 'T: 0 (homo)'],
             ['T: 0 (hetero)', 'T: 20 (hetero)']]

num_items_tasks = [[8, 16, 32],
                   [1, 6, 12],
                   [3, 6, 9],
                   [3, 6, 9],
                   [1, 4, 8, 12],
                   [1, 4, 8, 12]]

num_stimuli_per_case_tasks = [30, 30, 36, 36, 30, 30]
margin_map = {0: 0, 1: 2, 2: 1}

color_data = {}
# Categorization
color_data['T: 0 (homo)'] = 1
color_data['T: 20 (homo)'] = 0
color_data['T: 0 (hetero)'] = 0
color_data['T: 20 (hetero)'] = 1

# Curvature
color_data['Curve'] = 0
color_data['Line'] = 1

# Intersection
color_data['No Intersection'] = 0
color_data['Cross'] = 1
color_data['T'] = 0
color_data['L'] = 1

# Lighting Direction
color_data['Horizontal'] = 1
color_data['Vertical'] = 0

class FxnData:
    def __init__(self, exp_name, exp_task, num_fxn, num_item, rt_hum, num_item_rt, model_name):
        self.name = exp_name
        self.task = exp_task
        self.num_fxn = num_fxn
        self.num_item = num_item
        self.rt_hum = np.array(rt_hum)
        self.num_item_rt = num_item_rt
        self.model_name = model_name

    def calcModelRT(self, rt_fxn_model):
        M = rt_fxn_model['slope']
        C = rt_fxn_model['intercept']

        rt_model = []
        rt_model_std = []
        for item in self.num_item_rt:
            rt_model.append(M*(np.mean(self.num_fxn[self.num_item==item]))+C)
            rt_model_std.append(M*(np.std(self.num_fxn[self.num_item==item]))/np.sqrt(len(self.num_fxn[self.num_item==item])))

        self.rt_model = np.array(rt_model)
        self.rt_model_std = np.array(rt_model_std)

        self.model_slope, self.model_intercept, r_value, p_value, std_err = stats.linregress(self.num_item_rt, self.rt_model)
        self.hum_slope, self.hum_intercept, r_value, p_value, std_err = stats.linregress(self.num_item_rt, self.rt_hum)

def get_fxnVSitem(exp_name, exp_task, data_path, data_type, num_items, num_stimuli_per_case, model_name, rt_fxn_model):
    data = np.genfromtxt(data_path, delimiter=',')
    num_fxn = np.zeros(num_stimuli_per_case*len(num_items))
    num_item = np.zeros(num_stimuli_per_case*len(num_items))

    for j in range(len(num_items)):
        for k in range(1, num_stimuli_per_case+1):
            num_fxn[j*num_stimuli_per_case + k-1] = np.argmax(data[j*num_stimuli_per_case + k-1,:])
            num_item[j*num_stimuli_per_case + k-1] = num_items[j]
            if np.sum(data[j*num_stimuli_per_case + k-1, :])==0:
                num_fxn[j*num_stimuli_per_case + k-1] = np.max(num_items[j])

    rt_hum = human_rt_data[data_type]
    fxn_data = FxnData(exp_name, exp_task, num_fxn, num_item, rt_hum, num_items, model_name)
    fxn_data.calcModelRT(rt_fxn_model)

    return fxn_data

def gen_RT_data(method, method_dir, rt_fxn_model, task_ids):
    all_fxn_data = []
    all_fxn_seq = []
    for n in task_ids:
        data_type = file_names[n]
        temp = []
        for i in range(len(data_type)):
            exp_name = exp_names[n]
            exp_task = exp_tasks[n][i]
            num_items = np.array(num_items_tasks[n])
            num_stimuli_per_case = num_stimuli_per_case_tasks[n]
            data_path = method_dir + method + "/out_data/" + tasks[n] + "_" + data_type[i] + ".csv"
            all_fxn_data.append(get_fxnVSitem(exp_name, exp_task, data_path, data_type[i], num_items, num_stimuli_per_case, method, rt_fxn_model))

            temp.append(len(all_fxn_data)-1)

        all_fxn_seq.append(temp)

    return all_fxn_data, all_fxn_seq

def gen_RT_plot(all_fxn_data, task_ids, n, save_subdir="./"):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['mathtext.fontset'] = 'cm'

    fig = plt.figure()
    fig.set_size_inches(4, 1.2)

    ax2 = plt.subplot(121)
    plt.ylabel("RT (ms)")
    plt.xlabel("Number of items")

    ax1 = plt.subplot(122)
    plt.xlabel("Number of items")
    # ax1.set_title("Model")
    # ax2.set_title("Human")

    maxy = 0
    miny = 10000

    for i, taskid in enumerate(task_ids):
        fxn_data = all_fxn_data[taskid]
        if maxy < np.max(fxn_data.rt_model):
            maxy = np.max(fxn_data.rt_model)

        if maxy < np.max(fxn_data.rt_hum):
            maxy = np.max(fxn_data.rt_hum)

        if miny > np.min(fxn_data.rt_model):
            miny = np.min(fxn_data.rt_model)

        if miny > np.min(fxn_data.rt_hum):
            miny = np.min(fxn_data.rt_hum)

        ax1.errorbar(fxn_data.num_item_rt, fxn_data.rt_model+0*margin_map[i], yerr=fxn_data.rt_model_std, c=colors[i], label=fxn_data.task, fmt='--')
        ax2.plot(fxn_data.num_item_rt, fxn_data.rt_hum+0*margin_map[i], c=colors[i], label=fxn_data.task)

    ax1.set_ylim(miny-100, maxy+50)
    ax2.set_ylim(miny-100, maxy+50)

    ax1.set_xticks(all_fxn_data[task_ids[0]].num_item_rt)
    ax2.set_xticks(all_fxn_data[task_ids[0]].num_item_rt)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax2.legend(loc='upper left', bbox_to_anchor=(-0.55, 1.5), frameon=False, ncol=3)
    plt.subplots_adjust(wspace=0.45)

    try:
        os.makedirs("results/rt_plots/" + all_fxn_data[task_ids[0]].model_name)
    except:
        pass

    fig.savefig(save_subdir + "results/rt_plots/" + all_fxn_data[task_ids[0]].model_name + "/" + str(n) + ".pdf", dpi=150, bbox_inches="tight")
    plt.close()
