import numpy as np
from rt_plot import gen_RT_data
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import copy
from scipy import stats
import matplotlib as mpl

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

def calcAsymScore(all_fxn_data, all_fxn_seq, human=False):
    aT = []
    expNames = []
    for i in range(len(all_fxn_seq)):
        task_id = all_fxn_seq[i]

        if all_fxn_data[task_id[0]].hum_slope > all_fxn_data[task_id[1]].hum_slope:
            tH = task_id[0]
            tL = task_id[1]
        else:
            tL = task_id[0]
            tH = task_id[1]

        if human:
            sH, sL = max(0, all_fxn_data[tH].hum_slope), max(0, all_fxn_data[tL].hum_slope)
        else:
            sH, sL = max(0, all_fxn_data[tH].model_slope), max(0, all_fxn_data[tL].model_slope)

        if sH + sL == 0:
            aT.append(0)
            expNames.append(all_fxn_data[tH].task)
        else:
            aT.append((sH-sL)/(sH+sL))
            expNames.append(all_fxn_data[tH].task)

    aT = np.array(aT)
    expNames = np.array(expNames)

    return aT, expNames

markers_data = ['o', 's', 'X', 'P', 'v', 'D']

def plotAsymScore(all_fxn_data, all_fxn_seq):
    aM, expNames = calcAsymScore(all_fxn_data, all_fxn_seq)
    aH, expNames = calcAsymScore(all_fxn_data, all_fxn_seq, True)

    plt.figure(figsize=(4.125/1.2, 4.125/1.2))
    ax = plt.subplot(111)

    for i in range(len(aH)):
        plt.scatter(aH[i], aM[i], c='k', label="Exp"+str(i+1), marker=markers_data[i])

    plt.plot([-1, 1], [0, 0], "--", c="gray")
    plt.plot([0, 0], [-1, 1], "--", c="gray")
    plt.axis('square')
    plt.axis('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)

    locs = np.arange(-1, 1.1, 0.5)
    plt.yticks(locs)
    plt.xticks(locs)

    plt.xlabel("Human Asymmetry Index")
    plt.ylabel("Model Asymmetry Index")
    
    plt.title("Average Asymmetry Index = " + str(np.round(np.mean(aM), 3)), loc='right', fontsize=9)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    try:
        os.makedirs("results/asym_index_scatter_plots")
    except:
        pass

    plt.savefig("results/asym_index_scatter_plots/"+all_fxn_data[0].model_name+".pdf", dpi=200, bbox_inches="tight")
    plt.close()

def calcCorrScore(all_fxn_data, all_fxn_seq):
    m_s = []
    h_s = []
    returndata = {}

    task_list = []
    for n in range(len(all_fxn_seq)):
        for j in all_fxn_seq[n]:
            if all_fxn_data[j].task in task_list:
                pass
            else:
                m_s.append(all_fxn_data[j].model_slope)
                h_s.append(all_fxn_data[j].hum_slope)
                task_list.append(all_fxn_data[j].task)

    R, P = stats.pearsonr(np.array(h_s), np.array(m_s))

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(h_s), np.array(m_s))
    R2 = r2_score(np.array(h_s), np.array(m_s))

    returndata['R'] = round(R, 4)
    returndata['P'] = round(P, 6)

    returndata['slope'] = round(slope, 2)
    returndata['intercept'] = round(intercept, 2)
    returndata['r_value'] = round(r_value, 4)
    returndata['p_value'] = round(p_value, 6)
    returndata['std_err'] = round(std_err, 6)

    return returndata
