#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import random
import math
from itertools import izip


model_filename = "run1.pickle"
config_filename = 'config.yml'
data_path = '/home/eavsteen/seizure_detection/data'
target_gpu = 'gpu0'
user = 'patient1'
common_args = ['--debug-sub-ratio=1', '--model-filename='+model_filename,'--config-filename='+config_filename, '--no-channels=1', '--target-gpu='+target_gpu, '--data-path='+data_path, '--patients='+user]



def train(ch):
	spec_args = ['--channels='+str(ch)]
	cmd = ['python', 'train_batch.py'] + common_args + spec_args
	print cmd
	sys.stdout.flush()
	subprocess.call(cmd)


for i in range(16):
	train(i)

