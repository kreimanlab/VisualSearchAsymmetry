#!/usr/bin/env python
# coding: utf-8

def generateAsymIndex():
    import numpy as np
    from rt_plot import gen_RT_data
    import os
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    import copy
    from scipy import stats
    import matplotlib as mpl

    from eval_asym_metric import calcAsymScore, plotAsymScore

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    blk_col = '#000000'

    mpl.rc('axes', edgecolor=blk_col, labelcolor=blk_col)
    mpl.rc('xtick', color=blk_col)
    mpl.rc('ytick', color=blk_col)
    mpl.rc('text', color=blk_col)

    plt.rcParams.update({'font.size': 11})
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Arial'})
    mpl.rcParams['lines.linewidth'] = 1.3
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['xtick.major.width'] = 1.3
    mpl.rcParams['ytick.major.width'] = 1.3

    rt_fxn_model = {'slope': 252.359, 'intercept': 376.271}

    methods = np.array(os.listdir("./eccNET/out/")+["pixelMatch", "chance", "gbvs"])
    dirs = ['./eccNET/out/']*(len(methods)-3)+["./"]*3

    task_ids = [*range(6)]

    score_overall = []
    for n in tqdm(range(len(methods)), desc="Generating Asymmetry Index Scatter Plots..."):
        method = methods[n]
        if method[0] != '.':
            all_fxn_data, all_fxn_seq = gen_RT_data(method, dirs[n], rt_fxn_model, task_ids)
            score = calcAsymScore(all_fxn_data, all_fxn_seq)[0]
            # print(method+":", np.round(np.mean(score), 3))
            score_overall.append([method, np.round(np.mean(score), 3)])
            plotAsymScore(all_fxn_data, all_fxn_seq)

    score = calcAsymScore(all_fxn_data, all_fxn_seq, True)[0]
    # print("Humans"+":", np.round(np.mean(score), 3))
    score_overall.append(["Humans", np.round(np.mean(score), 3)])
    print("-----------------------------------------\n")

    model_scores = np.array(score_overall)
    score_idx = np.flip(np.argsort(np.array(model_scores[:, 1], dtype=np.float32)))
    model_scores = model_scores[score_idx]
    print(model_scores)

    plt.figure(figsize=(4.25/1.2, 4.25/1.5))
    ax = plt.subplot(111)

    N = model_scores.shape[0]
    idx = np.arange(0, 10*N, 10)

    ax.bar(idx, (np.array(model_scores[:, 1], dtype=np.float32)), 8, color=["#609486", "#00CC96"]+(N-2)*["#BDBDBD"], edgecolor=blk_col)
    plt.ylabel("Asymmetry Index")
    plt.xticks(idx, model_scores[:, 0], rotation=45, position=(0.0, 0.01), va="top", ha="right", color=blk_col)
    plt.yticks(np.arange(-0.6, 0.65, 0.2))

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.savefig("results/asymIndexBarPlot.pdf", dpi=200, bbox_inches="tight")
    plt.close()

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

    methods = np.array(os.listdir("./eccNET/out/")+["pixelMatch", "chance", "gbvs"])
    dirs = ['./eccNET/out/']*(len(methods)-3)+["./"]*3
    task_ids = [*range(6)]

    for n in tqdm(range(len(methods)), desc="Generating RT Plots..."):
        method = methods[n]
        if method[0] != '.':
            all_fxn_data, all_fxn_seq = gen_RT_data(method, dirs[n], rt_fxn_model, task_ids)

            for n in range(len(all_fxn_seq)):
                gen_RT_plot(all_fxn_data, all_fxn_seq[n], task_ids[n])

def generateScatterPlot():
    import numpy as np
    from scatter_plot import createScatterPlots
    import os
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Arial'})

    rt_fxn_model = {'slope': 252.359, 'intercept': 376.271}
    task_ids = [*range(6)]
    len(task_ids)

    methods = np.array(os.listdir("./eccNET/out/")+["pixelMatch", "chance", "gbvs"])
    dirs = ['./eccNET/out/']*(len(methods)-3)+["./"]*3

    for n in tqdm(range(len(methods)), desc="Generating Slope Scatter Plots..."):
        method = methods[n]
        if method[0] != '.':
            corr = createScatterPlots(method, dirs[n], rt_fxn_model, task_ids, legends=True)

def RT_Significance_Test():
    from urllib.request import urlopen
    import numpy as np
    import sys
    from rt_plot import gen_RT_data
    import os
    from tqdm.auto import tqdm
    import pandas as pd

    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    from statsmodels.graphics.api import interaction_plot, abline_plot
    from statsmodels.stats.anova import anova_lm

    border = "".join(100*["-"]) + "\n"

    rt_fxn_model = {'slope': 252.359, 'intercept': 376.271}
    task_ids = [*range(6)]

    testData = []

    methods = np.array(os.listdir("./eccNET/out/")+["pixelMatch", "chance", "gbvs"])
    dirs = ['./eccNET/out/']*(len(methods)-3)+["./"]*3
    for n in range(len(methods)):
        method = methods[n]
        all_fxn_data, all_fxn_seq = gen_RT_data(method, dirs[n], rt_fxn_model, task_ids)

        header = ["Model"]
        modelTestData = [method]
        for t in range(len(all_fxn_seq)):
            data = []

            for i in range(len(all_fxn_seq[t])):
                fxn = all_fxn_data[all_fxn_seq[t][i]]
                exp_name = fxn.name

                rt = fxn.num_fxn*rt_fxn_model['slope'] + rt_fxn_model['intercept']
                setsize = fxn.num_item

                for j in range(rt.shape[0]):
                    data.append([rt[j], setsize[j], i])

            df = pd.DataFrame(data, columns = ['RT', 'SetSize', 'Task'])

            min_lm1 = ols('RT ~ SetSize', data=df).fit() #same slope same intercept
            min_lm2 = ols('RT ~ SetSize * Task', data = df).fit() #diff slope diff intercept
            table = anova_lm(min_lm1, min_lm2)

            modelTestData.append(table['Pr(>F)'][1])
            header.append(exp_name.replace(",", " - "))

        if n==0:
            testData.append(header)

        testData.append(modelTestData)

    np.savetxt("results/significance_test.csv", testData, delimiter=',', fmt="%s")

print("\nEvaluating Models on Asymmetry Index...")
print("------------------------------------------------")
generateAsymIndex()

print("\nGenerating RT Plots...")
print("------------------------------------------------")
generate_rt_plots()

print("\nGenerating Slope Scatter Plots...")
print("------------------------------------------------")
generateScatterPlot()
