#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import random
import math
from itertools import izip


config_filename = 'config2.yml'
data_path = '/home/eavsteen/seizure_detection/data'
target_gpu = 'gpu2'
user = 'patient3'
common_args = ['--debug-sub-ratio=1', '--config-filename='+config_filename, '--no-channels=1', '--target-gpu='+target_gpu, '--data-path='+data_path]



def train(ch):
	spec_args = ['--patients='+user+'_'+str(ch)]
	cmd = ['python', 'train.py'] + common_args + spec_args
	print cmd
	sys.stdout.flush()
	subprocess.call(cmd)


for i in range(16):
	train(i)

