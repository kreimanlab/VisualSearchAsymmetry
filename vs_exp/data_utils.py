import numpy as np
import cv2

base_data_path_o = "./dataset/"
exps = ['categorization', 'curvature', 'intersection', 'lightning_dir']
exps_id = {'categorization': 0, 'curvature': 1, 'intersection': 2, 'lightning_dir': 3}

exp_tasks = [['homo/t0', 'homo/t20', 'hetero/t0', 'hetero/t20'],
            ['curve_in_lines', 'line_in_curves'],
            ['cross', 'non_cross', 'Ls', 'Ts'],
            ['left_right', 'top_down']]

exp_names = [['homo_t0', 'homo_t20', 'hetero_t0', 'hetero_t20'],
            ['curve_in_lines', 'line_in_curves'],
            ['cross', 'non_cross', 'Ls', 'Ts'],
            ['left_right', 'top_down']]

arr_size = [16, 36, 9, 16]
data_size = [120, 90, 108, 90]
back_fill = [255, 0, 255, 27]
reverse_img = [0, 0, 0, 0]
deg2px_l = [30, 30, 30, 30]
weight_pattern_l = ['nl', 'nl', 'nl', 'nl']

dog_size_l = [[[3, 1], [5, 3]],
            [[3, 1], [5, 3]],
            [[5, 3], [7, 5]],
            [[3, 0], [5, 1]]]

img_fmt = ['.jpg', '.jpg', '.jpg', '.jpg']

def get_data_paths(exp_type, task_id, i, base_data_path=None):
    if base_data_path == None:
    	base_data_path = base_data_path_o

    exp_id = exps_id[exp_type]
    stim_path = base_data_path + exps[exp_id] + '/' + exp_tasks[exp_id][task_id] + '/stimuli/' + str(i+1) + img_fmt[exp_id]
    tar_path = base_data_path + exps[exp_id] + '/' + exp_tasks[exp_id][task_id] + '/target/' + str(i+1) + img_fmt[exp_id]
    gt_path = base_data_path + exps[exp_id] + '/' + exp_tasks[exp_id][task_id] + '/gt/' + str(i+1) + img_fmt[exp_id]

    return stim_path, gt_path, tar_path

def get_exp_info(task, base_data_path=None):
    if base_data_path == None:
    	base_data_path = base_data_path_o

    exp_id = exps_id[task]
    gt_mask = np.load(base_data_path + task + "/gt_mask.npy")
    bg_value = back_fill[exp_id]
    rev_img_flag = reverse_img[exp_id]
    NumFix = arr_size[exp_id] + 1
    NumStimuli = data_size[exp_id]
    ior_size = 10
    fix = None
    num_task = len(exp_tasks[exp_id])

    stim_path, gt_path, tar_path = get_data_paths(task, 0, 0, base_data_path=base_data_path)
    stim_shape = cv2.imread(stim_path).shape
    tar_shape = cv2.imread(tar_path).shape
    eye_res = stim_shape[0]

    exp_name = exp_names[exp_id]
    deg2px = deg2px_l[exp_id]
    weight_pattern = weight_pattern_l[exp_id]

    dog_size = dog_size_l[exp_id]

    exp_info = {}
    exp_info['eye_res'] = eye_res
    exp_info['stim_shape'] = stim_shape
    exp_info['tar_shape'] = tar_shape
    exp_info['ior_size'] = ior_size
    exp_info['NumStimuli'] = NumStimuli
    exp_info['NumFix'] = NumFix
    exp_info['gt_mask'] = gt_mask
    exp_info['fix'] = fix
    exp_info['bg_value'] = bg_value
    exp_info['rev_img_flag'] = rev_img_flag
    exp_info['num_task'] = num_task
    exp_info['exp_name'] = exp_name
    exp_info['deg2px'] = deg2px
    exp_info['weight_pattern'] = weight_pattern
    exp_info['dog_size'] = dog_size

    return exp_info
