#!/usr/bin/env python
# coding: utf-8

import numpy as np

rt_hum = {}

# Categorization
pu = 200/9.5
rt_hum['homo_t0'] = [400+1.5*pu, 400+7*pu, 400+7.25*pu, 400+7.5*pu]
rt_hum['homo_t20'] = [400+0.7*pu, 400+1.8*pu, 400+1.95*pu, 400+2.1*pu]
rt_hum['hetero_t0'] = [400+1.5*pu, 400+8*pu, 600+1.6*pu, 600+5.1*pu]
rt_hum['hetero_t20'] = [400+6*pu, 800+1.5*pu, 1000+4*pu, 1200+6*pu]

# Curvature
rt_hum['curve_in_lines'] = [510, 510+8*0.8, 510+24*0.8]
rt_hum['line_in_curves'] = [565, 640, 750]

# Intersection
num_items = [3, 6, 9]
rt_hum['cross'] = 541 + 96*np.asarray(num_items)
rt_hum['non_cross'] = 510 + 45*np.asarray(num_items)
rt_hum['Ls'] = 474 + 33*np.asarray(num_items)
rt_hum['Ts'] = 499 + 14*np.asarray(num_items)

# Lighting Direction
rt_hum['left_right'] = [400 + 3.3*200/7.5, 600 + 2.8*200/7.5, 600 + 5.8*200/7.5]
rt_hum['top_down'] = [400+200/7.5, 400+1.2*200/7.5, 400+2.7*200/7.5]