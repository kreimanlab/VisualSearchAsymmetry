from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import ScanMatchPy
import matlab
from tqdm.notebook import tqdm
import matplotlib as mpl
import copy

NumStimuliDataset = {'ObjArr': 300, 'Waldo': 67, 'NaturalDesign': 240}
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

blk_col = '#212121'
mpl.rc('axes', edgecolor=blk_col, labelcolor=blk_col)
mpl.rc('xtick', color=blk_col)
mpl.rc('ytick', color=blk_col)
mpl.rc('text', color=blk_col)

def CummScore(dataset_name, model_names):
    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"

    CP_human = np.load(cp_dir + 'CP_' + 'human_all.npy')
    CP_human_err = np.std(CP_human, 0)/np.sqrt(CP_human.shape[0])
    CP_human = np.cumsum(np.mean(CP_human, 0))
    CP_human_sum = np.sum(CP_human)

    CP_IVSN = np.load(cp_dir + 'CP_' + 'IVSN.npy')
    CP_IVSN_err = np.std(CP_IVSN, 0)/np.sqrt(CP_IVSN.shape[0])
    CP_IVSN = np.cumsum(np.mean(CP_IVSN, 0))
    CP_IVSN_sum = np.sum(CP_IVSN)

    for i in range(len(model_names)):
        if model_names[i][0] == '.' or model_names[i][:5] == 'human' or model_names[i][:4] == 'IVSN':
            continue

        CP_model = np.load(cp_dir + 'CP_' + model_names[i])
        CP_model_err = np.std(CP_model, 0)/np.sqrt(CP_model.shape[0])
        CP_model = np.cumsum(np.mean(CP_model, 0))
        CP_model_sum = np.sum(CP_model)

        plt.figure()
        plt.errorbar(np.arange(1, CP_human.shape[0]), CP_human[1:], color=colors[1], yerr=CP_human_err[1:], label='Human')
        plt.errorbar(np.arange(1, CP_IVSN.shape[0]), CP_IVSN[1:], color='#919191', yerr=CP_IVSN_err[1:], label='IVSN')
        plt.errorbar(np.arange(1, CP_model.shape[0]), CP_model[1:], color=colors[2], yerr=CP_model_err[1:], label=model_names[i][:-4])

        # plt.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
        plt.legend(loc='lower right', frameon=False)
        plt.title(dataset_name)
        plt.xlabel('Fixation Number')
        plt.ylabel('Cummalative Performance')

        os.makedirs("results/" + dataset_name + "/CummScore/", exist_ok = True)
        plt.savefig("results/" + dataset_name + "/CummScore/" + model_names[i][:-4] + ".eps",  dpi=150)

def f2D(dataset_name, model_names):
    if dataset_name == "ObjArr":
        __f2D_ObjArr(dataset_name, model_names)
    else:
        __f2D(dataset_name, model_names)

def __f2D(dataset_name, model_names):
    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"
    tar_dir = "out/" + dataset_name + "/target_pos.npy"

    binranges = np.linspace(0.0, 30.0, num=61)
    col = colors[:6]
    lbls = ['L-0', 'L-1', 'L-2', 'L-3', 'L-4', 'L-5']

    for n in range(len(model_names)):
        if model_names[n][0] == '.':
            continue

        bincounts = []
        medians = []

        Saccades = np.load(fix_dir + model_names[n])
        I_model = np.load(ifix_dir + 'I_' + model_names[n])
        target_pos = np.load(tar_dir)

        NumStimuli = target_pos.shape[0]
        NumUnits = int(Saccades.shape[0]/NumStimuli)

        tf_i = I_model.reshape(-1,) - 1
        tf_n = [*range(Saccades.shape[0])]

        gt = np.zeros((NumUnits*NumStimuli, 2))
        for i in range(NumUnits):
            gt[i*NumStimuli:(i+1)*NumStimuli, :] = target_pos

        for lastn in range(6):
            lastind = tf_i - lastn
            pos = np.double(Saccades[tf_n, lastind, :])

            dist = (gt-pos)**2
            dist = np.sqrt(dist[:, 0] + dist[:, 1])*5/156
            dist = dist[lastind>0]

            hist = np.histogram(dist, binranges)
            bincounts.append(hist[0]/np.sum(hist[0]))
            medians.append(np.median(dist))

        plt.figure()
        for j in range(len(bincounts)):
            plt.plot(binranges[:60], bincounts[j], c=col[j], lw=1.2)

        for j in range(len(bincounts)):
            plt.plot([medians[j], medians[j]], [0, 0.3], '--', c=col[j], lw=1.2)

        plt.title('Model: ' + model_names[n][:-4] )
        plt.xlabel('Euclidean Distance To Target (visual degrees)')
        plt.ylabel('Proportion')
        # plt.legend(lbls, loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
        plt.legend(lbls, loc='upper right', frameon=False)
        # plt.savefig("results/" + dataset_name + "/fxn2targetDist/" + model_names[n][:-4] + ".png", dpi=300, bbox_inches="tight")
        os.makedirs("results/" + dataset_name + "/fxn2targetDist/", exist_ok = True)
        plt.savefig("results/" + dataset_name + "/fxn2targetDist/" + model_names[n][:-4] + ".eps", dpi=150)

def __f2D_ObjArr(dataset_name, model_names):
    xtrans = {640: 640, 365: 490, 90: 340, 915: 790, 1190: 940, 0:0}
    ytrans = {512: 512, 988: 772, 36: 252, 0:0}

    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"

    binranges = np.linspace(0.0, 30.0, num=61)
    col = colors[:6]
    lbls = ['L-0', 'L-1', 'L-2', 'L-3', 'L-4', 'L-5']

    for n in range(len(model_names)):
        if model_names[n][0] == '.':
            continue

        bincounts = []
        medians = []

        Saccades = np.load(fix_dir + model_names[n])
        for a1 in range(Saccades.shape[0]):
            for a2 in range(Saccades.shape[1]):
                Saccades[a1, a2, 0] = xtrans[Saccades[a1, a2, 0]]
                Saccades[a1, a2, 1] = ytrans[Saccades[a1, a2, 1]]

        CP_model = np.load(cp_dir + 'CP_' + model_names[n])
        I_model = np.load(ifix_dir + 'I_' + model_names[n])

        tf_i = I_model.reshape(-1,) - 1
        tf_n = [*range(Saccades.shape[0])]
        gt = np.double(Saccades[tf_n, tf_i, :])

        for lastn in range(6):
            lastind = tf_i - lastn
            pos = np.double(Saccades[tf_n, lastind, :])

            dist = (gt-pos)**2
            dist = np.sqrt(dist[:, 0] + dist[:, 1])*5/156
            dist = dist[lastind>0]

            hist = np.histogram(dist, binranges)
            bincounts.append(hist[0]/np.sum(hist[0]))
            medians.append(np.median(dist))

        plt.figure()
        for j in range(len(bincounts)):
            plt.plot(binranges[:60], bincounts[j], c=col[j], lw=0.8)

        for j in range(len(bincounts)):
            plt.plot([medians[j]+0.1*j, medians[j]+0.1*j], [0, 1], '--', c=col[j], lw=0.8)

        plt.title('Model: ' + model_names[n][:-4] )
        plt.xlabel('Euclidean Distance To Target (visual degrees)')
        plt.ylabel('Proportion')
        # plt.legend(lbls, loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
        plt.legend(lbls, loc='upper right', frameon=False)
        # plt.savefig("results/" + dataset_name + "/fxn2targetDist/" + model_names[n][:-4] + ".png", dpi=300, bbox_inches="tight")

        os.makedirs("results/" + dataset_name + "/fxn2targetDist/", exist_ok = True)
        plt.savefig("results/" + dataset_name + "/fxn2targetDist/" + model_names[n][:-4] + ".eps", dpi=150)

def SaccadeDist(dataset_name, model_names):
    binranges = np.linspace(0.0, 40.0, num=81)
    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"

    for i in range(len(model_names)):
        if model_names[i][0] == '.' or model_names[i][:5] == 'human' or model_names[i][:4] == 'IVSN':
            continue

        lbls = [model_names[i][:-4], 'IVSN', 'human_all']
        col = [colors[2], '#919191', colors[1]]

        plt.figure()
        for k in range(3):
            I_model = np.load(ifix_dir + 'I_' + lbls[k] + ".npy")
            Saccades = np.load(fix_dir + lbls[k] + ".npy")

            if dataset_name == "ObjArr":
                xtrans = {640: 640, 365: 490, 90: 340, 915: 790, 1190: 940, 0:0}
                ytrans = {512: 512, 988: 772, 36: 252, 0:0}
                for a1 in range(Saccades.shape[0]):
                    for a2 in range(Saccades.shape[1]):
                        Saccades[a1, a2, 0] = xtrans[Saccades[a1, a2, 0]]
                        Saccades[a1, a2, 1] = ytrans[Saccades[a1, a2, 1]]

            tf_k = I_model.reshape(-1,) - 1
            tf_n = [*range(Saccades.shape[0])]
            dist_t = np.asarray([])

            for j in range(np.max(tf_k)):
                tf_i = tf_k - j
                tf_j = tf_k - j - 1

                gt = np.double(Saccades[tf_n, tf_i, :])
                pos = np.double(Saccades[tf_n, tf_j, :])

                dist = (gt-pos)**2
                dist = np.sqrt(dist[:, 0] + dist[:, 1])*5/156
                dist = dist[tf_j>0]
                dist_t = np.concatenate((dist_t, dist))

            hist = np.histogram(dist_t, binranges)
            bincounts = hist[0].reshape(-1, 1)
            plt.subplot(311+k)
            plt.plot(binranges[:80], bincounts/np.sum(bincounts), c=col[k], lw=1.2, label=lbls[k])
            plt.legend(loc='upper right', frameon=False)
            if k == 1:
                plt.ylabel('Proportion')
            elif k == 0:
                plt.title('Model: ' + model_names[i][:-4] )

#         plt.title('Model: ' + model_names[i][:-4] )
        plt.xlabel('Saccade Sizes (visual degrees)')

        os.makedirs("results/" + dataset_name + "/SaccadeSize/", exist_ok = True)
        plt.savefig("results/" + dataset_name + "/SaccadeSize/" + model_names[i][:-4] + ".eps", dpi=150)

def ScanpathScore(dataset_name, model_names, focus):
    ss_dir = "out/" + dataset_name + "/ss_score/"
    try:
        os.makedirs(ss_dir)
    except:
        pass

    __ss_scoreHvH(dataset_name)
    __ss_score(dataset_name, model_names)

    calc_scores = os.listdir(ss_dir)

    if calc_scores[0] != 'human.npy':
        tmp = calc_scores[0]
        calc_scores.remove('human.npy')
        calc_scores.append(copy.deepcopy(tmp))
        calc_scores[0] = 'human.npy'

    if calc_scores[1] != model_names[focus]:
        tmp = calc_scores[1]
        calc_scores.remove(model_names[focus])
        calc_scores.append(copy.deepcopy(tmp))
        calc_scores[1] = model_names[focus]

    if calc_scores[2] != 'IVSN.npy':
        tmp = calc_scores[2]
        calc_scores.remove('IVSN.npy')
        calc_scores.append(copy.deepcopy(tmp))
        calc_scores[2] = 'IVSN.npy'

    model_scores = []
    rem_model = []

    for model_name in calc_scores:
        if (model_name[0] == '.'):
            rem_model.append(model_name)
        elif model_name in model_names or model_name[:5] == "human":
            model_scores.append(np.load(ss_dir+model_name))
        else:
            rem_model.append(model_name)

    for model_name in rem_model:
        calc_scores.remove(model_name)

    for n in range(len(calc_scores)):
        calc_scores[n] = calc_scores[n][:-4]

    model_scores = np.asarray(model_scores)

    os.makedirs("results/" + dataset_name + "/ScanPathScore/", exist_ok = True)

    plt.figure()
    N = model_scores.shape[0]
    idx = np.arange(0, 5*N, 5)
    plt.bar(idx, model_scores[:, 0], 3.5, yerr=model_scores[:, 1], color=tuple([colors[1], colors[2], '#919191'] + (len(model_scores[:, 0])-3)*['#BDBDBD']), edgecolor=blk_col)
    plt.ylabel('Scanpath Similarity Scores', labelpad=15)
    plt.xlabel('Models', labelpad=20)
    plt.title(dataset_name)
    plt.xticks(idx, calc_scores, rotation=90, position=(0.5, 0.42), va="center", ha="center", size='small', color=blk_col)
    plt.yticks(np.arange(0, np.max(model_scores[:, 0]) + 0.1, 0.1))
    plt.savefig("results/" + dataset_name + "/ScanPathScore/ScanPathScores.eps", dpi=150)

    score_save = np.array(calc_scores).reshape(-1, 1)
    score_save = np.concatenate((score_save.T, model_scores.T)).T
    np.savetxt("results/" + dataset_name + "/ScanPathScore/ScanPathScores.csv", score_save, delimiter=",", fmt="%s")

    print("Dataset:", dataset_name)
    for i in range(len(calc_scores)):
        spacing = ''
        for j in range(50 - len(calc_scores[i])):
            spacing += '_'

        print(calc_scores[i] + spacing, round(100*model_scores[i][0], 2))
    print()

def __ss_scoreHvH(dataset_name):
    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"
    ss_dir = "out/" + dataset_name + "/ss_score/"

    calc_scores = os.listdir(ss_dir)

    if 'human.npy' not in calc_scores:
        data_human = np.load(fix_dir + "human_all.npy")
        I_data_human = np.load(ifix_dir + "I_human_all.npy")
        NumStimuli = NumStimuliDataset[dataset_name]

        metric = ScanMatchPy.initialize()

        scores = []
        for i in tqdm(range(15), desc= dataset_name + " - human"):
            fix_model = data_human[i*NumStimuli:(i+1)*NumStimuli]
            fix1 = matlab.int32(np.int32(fix_model).tolist())
            I_fix1 = matlab.int32(np.int32(I_data_human[i*NumStimuli:(i+1)*NumStimuli]).tolist())

            for sub_id in range(i + 1, 15):
                fix_sub = data_human[sub_id*NumStimuli:(sub_id+1)*NumStimuli]
                fix2 = matlab.int32(np.int32(fix_sub).tolist())
                I_fix2 = matlab.int32(np.int32(I_data_human[sub_id*NumStimuli:(sub_id+1)*NumStimuli]).tolist())
                score = metric.findScore(fix1, fix2, I_fix1, I_fix2)
                scores.append(score)

        metric.terminate()

        scores = np.asarray(scores)

        f_score = np.mean(scores)
        f_std = np.std(scores)/np.sqrt(scores.shape[0])
        np.save(ss_dir + 'human.npy', [f_score, f_std])

def __ss_score(dataset_name, model_names):
    fix_dir = "out/" + dataset_name + "/fix/"
    cp_dir = "out/" + dataset_name + "/cp/"
    ifix_dir = "out/" + dataset_name + "/ifix/"
    ss_dir = "out/" + dataset_name + "/ss_score/"

    calc_scores = os.listdir(ss_dir)

    data_human = np.load(fix_dir + "human_all.npy")
    I_data_human = np.load(ifix_dir + "I_human_all.npy")
    NumStimuli = NumStimuliDataset[dataset_name]

    metric = ScanMatchPy.initialize()
    for i in range(len(model_names)):
        model_name = model_names[i]
        if (model_names[i][0] == '.') or (model_names[i][:5] == 'human') or (model_names[i] in calc_scores):
            continue

        fix_model = np.load(fix_dir + model_name)
        I_fix_model = np.load(ifix_dir + 'I_' + model_name)

        fix1 = matlab.int32(np.int32(fix_model).tolist())
        I_fix1 = matlab.int32(np.int32(I_fix_model).tolist())

        scores = []
        for sub_id in tqdm(range(15), desc=dataset_name + " - " + model_name[:-4]):
            fix_sub = data_human[sub_id*NumStimuli:(sub_id+1)*NumStimuli]
            fix2 = matlab.int32(np.int32(fix_sub).tolist())
            I_fix2 = matlab.int32(np.int32(I_data_human[sub_id*NumStimuli:(sub_id+1)*NumStimuli]).tolist())
            score = metric.findScore(fix1, fix2, I_fix1, I_fix2)
            scores.append(score)

        scores = np.asarray(scores)
        f_score = np.mean(scores)
        f_std = np.std(scores)/np.sqrt(scores.shape[0])

        np.save(ss_dir + model_name, [f_score, f_std])

    metric.terminate()
