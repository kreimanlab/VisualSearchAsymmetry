#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "../../")

import sys
sys.path.insert(0, "../")


# In[2]:


import numpy as np
import cv2
from vs_model import VisualSearchModel as VisualSearchModel
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from data_utils import get_data_paths, get_exp_info
import os
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import time

t_start = time.time()

base_data_path = "../dataset/"


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')

for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)


# In[3]:


# eccNET parameters

eccParam = {}
eccParam['rf_min'] = [2]*5
eccParam['stride'] = [2]*5
eccParam['ecc_slope'] = [0, 0, 3.5*0.02, 8*0.02, 16*0.02]
eccParam['deg2px'] = [round(30.0), round(30.0/2), round(30.0/4), round(30.0/8), round(30.0/16)]
eccParam['fovea_size'] = 4
eccParam['rf_quant'] = 1
eccParam['pool_type'] = 'avg'


# In[5]:


ecc_models = []

vgg_model_path = "../../pretrained_model/vgg16_imagenet_filters.h5"

# eccNET
for out_layer in [[1, 1, 1]]:
    model_desc = {'eccParam': eccParam,
                  'ecc_depth': 5,
                  'out_layer': out_layer,
                  'comp_layer': 'diff',
                  'vgg_model_path': vgg_model_path,
                  'model_subname': ""}

    ecc_models.append(model_desc)


# In[ ]:


exps = ['categorization', 'curvature', 'intersection', 'lightning_dir']

pbar = tqdm(total=len(ecc_models)*12)
for model_desc in ecc_models:
    vsm = VisualSearchModel(model_desc)
    print(vsm.model_name)

    save_path = "out/" + vsm.model_name
    try:
        os.makedirs(save_path + "/out_data/")
        os.makedirs(save_path + "/fix/")
    except:
        pass

    for exp_type in exps:
        exp_info = get_exp_info(exp_type, base_data_path=base_data_path)
        vsm.load_exp_info(exp_info, corner_bias=16*4*1)

        num_task = exp_info['num_task']
        NumStimuli = exp_info['NumStimuli']
        NumFix = exp_info['NumFix']
        exp_name = exp_info['exp_name']

        for task_id in range(num_task):
            data = np.zeros((NumStimuli, NumFix, 2))
            CP = np.zeros((NumStimuli, NumFix), dtype=int)

            for i in tqdm(range(NumStimuli), desc=exp_name[task_id]):
                stim_path, gt_path, tar_path = get_data_paths(exp_type, task_id, i, base_data_path=base_data_path)
                saccade = vsm.start_search(stim_path, tar_path, gt_path)

                j = saccade.shape[0]
                if j < NumFix+1:
                    CP[i, j-1] = 1
                data[i, :min(NumFix, j), 0] = saccade[:, 0].reshape((-1,))[:min(NumFix, j)]
                data[i, :min(NumFix, j), 1] = saccade[:, 1].reshape((-1,))[:min(NumFix, j)]

            file_name = save_path + '/out_data/' + exp_type + '_' + exp_name[task_id] + '.csv'
            np.savetxt(file_name, CP, delimiter=',', fmt="%d")
            file_name = save_path + '/fix/' + exp_type + '_' + exp_name[task_id] + '.npy'
            np.save(file_name, data)

            pbar.update(1)

pbar.close()

print("Total time taken:", time.time()-t_start)


# In[ ]:
