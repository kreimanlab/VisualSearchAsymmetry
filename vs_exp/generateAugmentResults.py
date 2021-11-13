#!/usr/bin/env python
# coding: utf-8

def generate_rt_plots():
    import numpy as np
    from rt_plot import gen_RT_data, gen_RT_plot
    import os
    from tqdm.auto import tqdm
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Arial'})

    mpl.rcParams['lines.linewidth'] = 1.3
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['xtick.major.width'] = 1.3
    mpl.rcParams['ytick.major.width'] = 1.3

    rt_fxn_model = {'slope': 252.359, 'intercept': 376.271}

    methods = np.array(os.listdir("./eccNET/out_aug/"))
    dirs = ['./eccNET/out_aug/']*(len(methods))
    task_ids = [*range(6)]

    for n in tqdm(range(len(methods)), desc="Generating RT Plots..."):
        method = methods[n]
        if method[0] != '.':
            all_fxn_data, all_fxn_seq = gen_RT_data(method, dirs[n], rt_fxn_model, task_ids)

            for n in range(len(all_fxn_seq)):
                gen_RT_plot(all_fxn_data, all_fxn_seq[n], task_ids[n])


print("\nGenerating RT Plots...")
print("------------------------------------------------")
generate_rt_plots()
