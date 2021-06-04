#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "../")


# In[2]:


import os
import numpy as np
import cv2
from tqdm.auto import tqdm
from utils import recog, get_act, remove_attn
from data_utils import get_data_paths, get_exp_info
import matplotlib.pyplot as plt

base_data_path = "../dataset/"

np.random.seed(1000)


# In[3]:


exps = ['categorization', 'curvature', 'intersection', 'lightning_dir']

num_exp = len(exps)
for exp_id in range(num_exp):
    exp_type = exps[exp_id]
    exp_info = get_exp_info(exp_type, base_data_path=base_data_path)
    num_task = exp_info['num_task']
    NumStimuli = exp_info['NumStimuli']
    NumFix = exp_info['NumFix']
    exp_name = exp_info['exp_name']
    bg_value = exp_info['bg_value']

    for task_id in range(num_task):
        # Load gt masks -----------------------------------------------------------
        gt_mask = np.load("../dataset/" + exps[exp_id] + "/gt_mask.npy")
        gt_sum = np.sum(gt_mask, axis=0)
        # Load gt masks -----------------------------------------------------------

        # Run visual serach -------------------------------------------------------
        score = np.uint8(np.zeros((NumStimuli, NumFix+1)))
        for i in tqdm(range(NumStimuli), desc=exp_name[task_id]):
            stim_path, gt_path, tar_path = get_data_paths(exp_type, task_id, i, base_data_path=base_data_path)

            gt = cv2.imread(gt_path, 0)
            stim_img = cv2.imread(stim_path, 1)
            tar_img = cv2.imread(tar_path, 1)

            if exp_info['rev_img_flag']==1:
                stim_img = 255 - stim_img
                tar_img = 255 - tar_img

            mask = np.uint8(np.ones(gt.shape))
            x = int(gt.shape[0]/2)
            y = int(gt.shape[0]/2)

            found = recog(x, y, gt)
            for j in range(NumFix):
                if found == 1:
                    score[i-1, j] = 1
                    break

                out = np.copy(mask*gt_sum)
                (x, y) = np.random.choice(out.shape[0]), np.random.choice(out.shape[1])
                while out[x, y] == 0 and (stim_img[x-5:x+5, y-5:y+5] == bg_value).all():
                    (x, y) = np.random.choice(out.shape[0]), np.random.choice(out.shape[1])


                found = recog(x, y, gt)
                mask = remove_attn(mask, x, y, gt_mask)

        # Run visual serach -------------------------------------------------------
        try:
            os.makedirs("out_data/")
        except:
            pass

        file_name = 'out_data/' + exp_type + '_' + exp_name[task_id] + '.csv'
        np.savetxt(file_name, score, delimiter=',', fmt="%d")


# In[ ]:
