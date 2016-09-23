import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
from utils import calcFFT
from matplotlib.ticker import AutoMinorLocator

import scipy.io as sio
import pandas as pd
from analyze import analyze, analyze_1h


# analyze('/home/eavsteen/seizure_detection/data/train_1/1_', '_0.mat','train_1', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/train_1/1_', '_1.mat','train_1', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/train_2/2_', '_0.mat','train_2', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/train_2/2_', '_1.mat','train_2', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/train_3/3_', '_0.mat','train_3', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/train_3/3_', '_1.mat','train_3', 1, 10)

# analyze('/home/eavsteen/seizure_detection/data/test_1/1_', '.mat','test_1', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/test_2/2_', '.mat','test_2', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/test_3/3_', '.mat','test_3', 1, 10)

analyze_1h('/home/eavsteen/seizure_detection/data/train_1/1_', '_0.mat','train_1', 1, 1152)
analyze_1h('/home/eavsteen/seizure_detection/data/train_1/1_', '_1.mat','train_1', 1, 150)
analyze_1h('/home/eavsteen/seizure_detection/data/train_2/2_', '_0.mat','train_2', 1, 2196)
analyze_1h('/home/eavsteen/seizure_detection/data/train_2/2_', '_1.mat','train_2', 1, 150)
analyze_1h('/home/eavsteen/seizure_detection/data/train_3/3_', '_0.mat','train_3', 1, 1152)
analyze_1h('/home/eavsteen/seizure_detection/data/train_3/3_', '_1.mat','train_3', 1, 150)
