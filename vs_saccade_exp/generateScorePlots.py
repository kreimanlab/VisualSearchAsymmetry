#!/usr/bin/env python
# coding: utf-8

# # Score Analysis

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
from score_analysis import CummScore, f2D, SaccadeDist, ScanpathScore


dataset_names = ["Waldo", "NaturalDesign", "ObjArr"]

for dataset_name in dataset_names:
    fix_dir = "out/" + dataset_name + "/fix/"
    model_names = sorted(os.listdir(fix_dir))

    ScanpathScore(dataset_name, model_names, focus=1)
    CummScore(dataset_name, model_names)
    SaccadeDist(dataset_name, model_names)
    f2D(dataset_name, model_names)
