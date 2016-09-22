#!/home/robo_external/.local/bin python
# -*- coding: utf-8 -*-

import sys
import os
import datetime
import time
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec

from tools.utils import calcFFT
from tools.analyze import read_data_1h

import numpy as np
import hickle as hkl
import pickle
import yaml
import math
import random
import platform
import argparse 
import gc
#from memory_profiler import profile

from sklearn.base import clone


import datasets



#https://docs.python.org/2/library/argparse.html
parser = argparse.ArgumentParser(description='Preprocess/Train/Validate all data.')
parser.add_argument('--data-path', dest='data_path', action='store', default="/home/eavsteen/seizure_detection/data",
                   help='top level path of data (default: /home/eavsteen/seizure_detection/data)')
parser.add_argument('--model-filename', dest='model_filename', action='store',
                   default="netSpec.pickle",
                   help='save/read the model parameters to/from the filename given (default: netSpec.pickle)')
parser.add_argument('--config-filename', dest='config_filename', action='store',
                   default="config.yml",
                   help='read the configuration parameters from the filename given (default: config.yml)')
parser.add_argument('--no-preprocessing', dest='no_preprocessing', action='store_true', default=False,
                   help='skip preprocessing. load preprocessed data from file (default: false)')
parser.add_argument('--no-save-preprocessed', dest='no_save_preprocessed', action='store_true', default=False,
                   help='don\'t save the data after preprocessing. saves some time (default: false)')
parser.add_argument('--no-training', dest='no_training', action='store_true',  default=False,
                   help='skip training. load trained net from file (default: false)')
parser.add_argument('--no-shuffle-before-split', dest='shuffle_before_split', action='store_false',  default=True,
                   help='No random shuffle before split between train and validation set. (default: true)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true', default=False,
                   help='don\'t save the model after training. saves some time (default: false)')
parser.add_argument('--fixed-seed', dest='fixed_seed', action='store_true', default=True,
                   help='fixed random seed. (default: false)')
parser.add_argument('--plot-prob-dist', dest='plot_prob_dist', action='store_true', default=False,
                   help='Plot the distribution of the predicted probabilities for both wrongly and rightly predicted samples. (default: false)')
parser.add_argument('--exclude-user', dest='exclude_user', action='append', default=[],
					help='exclude data from specific user')
parser.add_argument('--include-user', dest='include_user', action='append', default=[],
					help='include only data from specific user')
parser.add_argument('--debug-sub-ratio', dest='debug_sub_ratio', action='store', type=float, default=1,
					help='use only a fraction of the data, e.g. 0.5, for faster experiments during debugging (default 1)')
parser.add_argument('--validation-ratio', dest='chosen_validation_ratio', action='store', type=float, default=0.2,
					help='validation ratio (default 0.2)')
parser.add_argument('--shift', dest='shift', action='store', type=int, default=0,
					help='Only at test time! Shift the window around the peak and predict for each shifted sample and add the probabilities. (default 0)')
parser.add_argument('--no-channels', dest='no_channels', action='store', type=int, default=16,
					help='The number of channels in the data (default: 4)')
parser.add_argument('--target-gpu',dest='target_gpu', action='store', default="gpu0",
                   help='target gpu')
parser.add_argument('--mode', dest='mode', action='store', default="None",
                   help='single-channel/dual-channel/none (default: none)')
args = parser.parse_args()

print "Command line arguments:", args
print "Git reference: ",
os.system("git show-ref HEAD")
print "Timestamp:", datetime.datetime.now()
print "Hostname:", platform.node()

import theano.sandbox.cuda
print args.target_gpu
theano.sandbox.cuda.use(args.target_gpu)
import lasagne

from convnets.processData import processDataSpectrum

if args.fixed_seed:
	random.seed(0) 
	np.random.seed(0)

#Read in and print parameters from config file
with open(args.config_filename, 'r') as ymlfile:
	print "Configuration %r:" % args.config_filename
	print ymlfile.read()
	print "end Configuration"
	ymlfile.seek(0)
	cfg = yaml.load(ymlfile)

sys.stdout.flush()

preprocess_params = cfg['preprocess']
floor = preprocess_params['floor']
ceil = preprocess_params['ceil']
fft_width = preprocess_params['fft_width']
overlap = preprocess_params['overlap']
magnitude_window = preprocess_params['magnitude_window']
include_userdata = preprocess_params['include_userdata']

height=fft_width/2
assert ceil-floor <= fft_width / 2
assert ceil <= fft_width / 2

# def analyze_record_quality_one_channel(data):
# 	global c_sat
# 	count_saturated = np.count_nonzero(data > (2**16*0.99)) or np.count_nonzero(data < (2**16*0.01))
# 	ratio_saturated = count_saturated / float(data.shape[0])
# 	# print 'count saturated: %g'%count_saturated
# 	if ratio_saturated > 0.05:
# 		# print 'ratio saturated: %g'%ratio_saturated
# 		c_sat += 1
# 		# print 'c_sat', c_sat
# 		return True
# 	return False

# def analyze_record_quality(samples, magnitudes):
# 	global r_sat
# 	sat = any(map(analyze_record_quality_one_channel, samples))
# 	if sat:
# 		r_sat += 1
# 		# print 'r_sat', r_sat
# 	return np.array([(sat,)], dtype=analysis_data_type)

def read_data(dataset,n_start,n_stop,s_start,s_stop,no_samples_normal_ph,no_samples_seizure_ph):
	global magnitudes
	global train_counter
	global val_counter

	"read data and preprocess (fft and slicing)"
	
	path = data_path+'/'+dataset.set_name+'/'+dataset.base_name
	print path

	# no_normal = int(dataset.no_normal * args.debug_sub_ratio)
	# no_seizure = int(dataset.no_seizure * args.debug_sub_ratio)
	# print "no_normal", no_normal
	# print "no_seizure", no_seizure

	# print x_size
	# no_samples_normal_ph = x_size/2/no_normal
	# print "no_samples_normal_ph", no_samples_normal_ph	


	# read in normal 
	is_train_index = get_train_val_split(n_stop)
	for i in xrange(n_start,n_stop):
		print "i", i
		sys.stdout.flush()
		data_1h = read_data_1h(path,'_0.mat',i*6+1)
		if is_train_index[i]:
			counter = train_counter
		else:
			counter = val_counter
		for ch in range(args.no_channels):
			magnitude = calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil]
			step = magnitude.shape[0]/no_samples_normal_ph
			assert no_samples_normal_ph < magnitude.shape[0]
			r = np.arange(0,magnitude.shape[0],step)
			r=r[0:no_samples_normal_ph]
			for j in range(no_samples_normal_ph):
				if is_train_index[i]:
					magnitudes[counter+j,ch] = magnitude[r[j]]
					labels[counter+j] = 0
				else:
					magnitudes[counter-j,ch] = magnitude[r[j]]
					labels[counter-j] = 0
				
		if is_train_index[i]:
			train_counter += no_samples_normal_ph
		else:
			val_counter -= no_samples_normal_ph

	# no_samples_seizure_ph = x_size/2/no_seizure
	# print "no_samples_seizure_ph", no_samples_seizure_ph	

	# read in seizure 
	is_train_index = get_train_val_split(s_stop)
	for i in xrange(s_start,s_stop):
		print "i", i
		sys.stdout.flush()
		data_1h = read_data_1h(path,'_1.mat',i*6+1)
		if is_train_index[i]:
			counter = train_counter
		else:
			counter = val_counter
		for ch in range(args.no_channels):
			magnitude = calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil]
			step = magnitude.shape[0]/no_samples_seizure_ph
			assert no_samples_seizure_ph < magnitude.shape[0]
			r = np.arange(0,magnitude.shape[0],step)
			r=r[0:no_samples_seizure_ph]
			for j in range(no_samples_seizure_ph):
				if is_train_index[i]:
					magnitudes[counter+j,ch] = magnitude[r[j]]
					labels[counter+j] = 1
				else:
					magnitudes[counter-j,ch] = magnitude[r[j]]
					labels[counter-j] = 1
		if is_train_index[i]:
			train_counter += no_samples_seizure_ph
		else:
			val_counter -= no_samples_seizure_ph

	print "counter", counter
	
	print "Done reading in", n_stop-n_start, "no seizure hours and", s_stop-s_start, "seizure hours"

def get_train_val_split(no):
	if args.fixed_seed:
		random.seed(0) 
		np.random.seed(0)
	all_indices = np.arange(no)
	print all_indices
	if args.shuffle_before_split:
		np.random.shuffle(all_indices)
	train_no = int(math.floor((1-args.chosen_validation_ratio)*no))
	print train_no
	train_file_indices = all_indices[:train_no]
	is_train_index = np.zeros(no, dtype=np.bool)
	print train_file_indices
	is_train_index[train_file_indices] = True
	# val_no = no - train_no
	# val_indices = all_indices[train_no:]
	return is_train_index

def process_taps(raw_magnitudes, raw_labels, raw_analysis_datas, session):
	global train_counter
	global val_counter

	tap_no_patches_ps = session.no_patches_ps

	len_sample = raw_magnitudes.shape[2]

	sos = np.sum(raw_magnitudes,axis=(0, 1, 3))
	max_pos = np.argmax(sos)
	# print "Max pos: %d in %d time slices"%(max_pos,sos.shape[0])

	is_train_index = get_train_val_split(len(raw_magnitudes))


	if args.no_training and args.shift>0:
		for i in xrange(len(raw_magnitudes)):
			energy_in_function_of_time = np.sum(raw_magnitudes[i], axis=(0, 2))
			local_max_pos = np.rint(np.sum(energy_in_function_of_time * np.arange(energy_in_function_of_time.shape[0]))/ np.sum(energy_in_function_of_time))
			offset = (args.shift-1)/2
			if is_train_index[i]:
				counter = train_counter
				train_counter += 1
			else:
				counter = val_counter
				val_counter -= 1
			magnitudes[counter] = raw_magnitudes[i,:,local_max_pos-magnitude_window/2-offset:local_max_pos+magnitude_window/2+offset,floor:ceil]
			labels[counter] = raw_labels[i]
			analysis_datas[:][counter] = raw_analysis_datas[:][i]
			userdata[counter][dict_all_users[session.user.lower()]]=1
	elif 'shift' in cfg['preprocess']['augmentation']:
		for i in xrange(len(raw_magnitudes)):
			#tap_no_patches_ps_on = tap_no_patches_ps*7 / 8
			#tap_no_patches_ps_off = tap_no_patches_ps - tap_no_patches_ps_on
			tap_no_patches_ps_on = tap_no_patches_ps
			tap_no_patches_ps_off = 0

			energy_in_function_of_time = np.sum(raw_magnitudes[i], axis=(0, 2))
			local_max_pos = np.rint(np.sum(energy_in_function_of_time * np.arange(energy_in_function_of_time.shape[0]))/ np.sum(energy_in_function_of_time))

			# if i == 1:
			# 	print "tap_no_patches_ps_on", tap_no_patches_ps_on
			# 	preprocessing "tap_no_patches_ps_off", tap_no_patches_ps_off
			local_on_condition = lambda offset:on_condition(local_max_pos - offset)
			local_off_condition = lambda offset:off_condition(local_max_pos - offset)
			r = range(0,len_sample-magnitude_window+1)
			r_on = filter(local_on_condition, r)
			r_off = filter(local_off_condition, r)
			offsets = np.concatenate([random.sample(r_on, tap_no_patches_ps_on), random.sample(r_off, tap_no_patches_ps_off)])
			offsets = offsets.astype(int)
			for k in range(tap_no_patches_ps):
				if is_train_index[i]:
					counter = train_counter
					train_counter += 1
				else:
					counter = val_counter
					val_counter -= 1

				magnitudes[counter] = raw_magnitudes[i,:,offsets[k]:offsets[k]+magnitude_window,floor:ceil]
				labels[counter] = [label_values.noise, raw_labels[i]][local_on_condition(offsets[k])]
				analysis_datas[:][counter] = raw_analysis_datas[:][i]
				if include_userdata:
					userdata[counter][dict_all_users[session.user.lower()]]=1
	else:
		for i in xrange(len(raw_magnitudes)):
			energy_in_function_of_time = np.sum(raw_magnitudes[i], axis=(0, 2))
			local_max_pos = np.rint(np.sum(energy_in_function_of_time * np.arange(energy_in_function_of_time.shape[0]))/ np.sum(energy_in_function_of_time))
			if is_train_index[i]:
				counter = train_counter
				train_counter += 1
			else:
				counter = val_counter
				val_counter -= 1
			magnitudes[counter] = raw_magnitudes[i,:,local_max_pos-magnitude_window/2:local_max_pos+magnitude_window/2,floor:ceil]
			labels[counter] = raw_labels[i]
			analysis_datas[:][counter] = raw_analysis_datas[:][i]
			userdata[counter][dict_all_users[session.user.lower()]]=1

def preprocess():
	global size
	global xTrain
	global udTrain
	global yTrain
	global aTrain
	global xVal
	global udVal
	global yVal
	global aVal
	global magnitudes
	global userdata
	global labels
	global analysis_datas

	print("Loading and preprocessing data...")

	multiplier = 4
	no_normal = int(datasets.patient0.no_normal * args.debug_sub_ratio)
	no_seizure = int(datasets.patient0.no_seizure * args.debug_sub_ratio)
	no_samples_normal_ph = multiplier * no_seizure
	no_samples_seizure_ph = multiplier * no_normal

	print "no_normal", no_normal
	print "no_seizure", no_seizure
	print "no_samples_normal_ph", no_samples_normal_ph
	print "no_samples_seizure_ph", no_samples_seizure_ph

	size = no_normal * no_samples_normal_ph + no_seizure * no_samples_seizure_ph

	print "size", size

	if args.no_training and args.shift > 0:
		magnitudes = np.zeros((size,args.no_channels,magnitude_window+args.shift-1,ceil-floor), dtype=np.float32)
	else:
		magnitudes = np.zeros((size,args.no_channels,magnitude_window,ceil-floor), dtype=np.float32)

	print "include_userdata", include_userdata
	if include_userdata:
		print "initializing userdata"
		userdata = np.zeros((size,no_users), dtype=np.int32)
	
	labels = np.zeros(size)
	# analysis_datas = np.zeros(size, dtype=analysis_data_type)

	global train_counter
	global val_counter
	train_counter = 0
	val_counter = size - 1

	dss = [datasets.patient0]
	no_dss = len(dss)

	for dataset in dss:
		print "Read in dataset from %s ..."%(dataset.set_name)
		print "Processing data ..."
	 	
	 	train_c_before = train_counter
		val_c_before = val_counter
		
		read_data(dataset,0,no_normal,0,no_seizure,no_samples_normal_ph,no_samples_seizure_ph)

		train_c_after = train_counter
		val_c_after = val_counter

	print 'train_counter', train_counter, 'val_counter', val_counter, 'size', size,
	assert val_counter == train_counter-1
	assert magnitudes.shape[0] == labels.shape[0]

	labels = labels.astype(np.int32)
	magnitudes = magnitudes.astype(np.float32)

	print("Histogram:")
	print np.bincount(labels)

	print "magnitudes.shape", magnitudes.shape
	print "labels.shape", labels.shape

	no_train = train_counter
	no_val = size-val_counter-1
	assert no_train + no_val == size
	print 'Ratio validation:', no_val/float(size)
	if abs(no_val/float(size) - args.chosen_validation_ratio) > 0.02:
		print "WARNING: validation ratio (%g) differs from expected value (%g)"%(no_val/float(size), args.chosen_validation_ratio)
	
	xTrain = magnitudes[:no_train]
	udTrain = []
	if include_userdata:
		udTrain = userdata[:no_train]
	yTrain = labels[:no_train]

	xVal = magnitudes[no_train:]
	udVal = []
	if include_userdata:
		udVal = userdata[no_train:]
	yVal = labels[no_train:]
	#aVal = analysis_datas[no_train:]


	print("Shuffling data...")
	a = np.arange(xTrain.shape[0])
	np.random.shuffle(a)
	xTrain = xTrain[a]
	if include_userdata:
		udTrain = udTrain[a]
	yTrain = yTrain[a]

	# inorder to be able to release magnitudes array
	xVal = np.copy(xVal)

	del magnitudes
	gc.collect()


	print 'xTrain.shape', xTrain.shape
	print 'yTrain.shape', yTrain.shape
	print 'xVal.shape', xVal.shape
	print 'yVal.shape', yVal.shape
	assert xTrain.shape[0] == yTrain.shape[0]
	assert xVal.shape[0] == yVal.shape[0]

	if not args.no_save_preprocessed:
		print("Saving preprocessed data...")
		data = {
			'xTrain':xTrain, 
			#'udTrain':udTrain, 
			#'aTrain':aTrain, 
			'yTrain':yTrain, 
			'xVal':xVal,
			#'udVal':udVal, 
			'yVal':yVal,
			}
		hkl.dump(data, 'preprocessedData.hkl',compression="lzf")

def load_preprocessed():
	#global include_userdata
	global xTrain
	global yTrain
	#global aTrain
	global xVal
	global yVal
	#global aVal
	global udTrain
	global udVal

	print("Loading preprocessed data....")
	data = hkl.load('preprocessedData.hkl')
	xTrain = data['xTrain']
	yTrain = data['yTrain']
	#aTrain = data['aTrain']
	xVal = data['xVal']
	yVal = data['yVal']
	#aVal = data['aVal']
	# if include_userdata:
	# 	udTrain = data['udTrain']
	# 	udVal = data['udVal']	

#@profile
def normalize_and_train(netSpec):
	global xTrain
	global xVal

	normalization_data = dict()
	print "Normalizing values "

	# xTrain = np.log(xTrain)
	# xVal = np.log(xVal)
	# maximum = np.amax(xTrain, keepdims=True)
	# print "Normalizing with log(x)/maximum*2-1 ", maximum
	# xTrain = xTrain/maximum*2.0-1.0
	# xVal = xVal/maximum*2.0-1.0
	# normalization_data['maximum'] = maximum

	# xTrain = np.log(xTrain)
	# xVal = np.log(xVal)
	# mean = np.mean(xTrain, keepdims=True)
	# stdev = np.std(xTrain-mean, keepdims=True)
	# print "Normalizing with mean ", mean, " stdev ", stdev
	# xTrain = (xTrain-mean)/stdev
	# xVal = (xVal-mean)/stdev
	# normalization_data['mean'] = mean
	# normalization_data['stdev'] = stdev

	if cfg['preprocess']['normalization'] == 'min_max_x255':
		print "percentiles:"
		for p in range(0,101,10):
			print p, np.percentile(xTrain, p)
		xTrain_mask = np.ma.masked_equal(xTrain,0.)
		maximum = np.amax(xTrain_mask, keepdims=True)
		minimum = np.amin(xTrain_mask, keepdims=True)
		print "Normalizing with ", minimum, maximum
		xTrain[xTrain>0.1] = (xTrain[xTrain>0.1]-minimum)/(maximum-minimum)*255.0
		xVal[xVal>0.1] = (xVal[xVal>0.1]-minimum)/(maximum-minimum)*255.0
		normalization_data['maximum'] = maximum
		normalization_data['minimum'] = minimum

	# referrers = gc.get_referrers(magnitudes)
	# for referrer in referrers:
	# 	print referrer.__name__, globals().__name__
	gc.collect()

	if cfg['preprocess']['normalization'] == 'log':
		print "Normalizing  log(1+x)*100 "
		xTrain = np.log10(1+xTrain)*100
		xVal = np.log10(1+xVal)*100

	# stdev = np.std(xTrain, keepdims=True)
	# mean = np.mean(xTrain, keepdims=True)
	# print "Normalizing with ", mean, stdev
	# xTrain = (xTrain-mean)*stdev
	# xVal = (xVal-mean)*stdev
	# normalization_data['mean'] = mean
	# normalization_data['stdev'] = stdev

	# percentile90 = np.percentile(xTrain,90, keepdims=True)
	# print "Normalizing with percentile90 ", percentile90
	# xTrain = xTrain/percentile90*255
	# xVal = xVal/percentile90*255
	# normalization_data['percentile90'] = percentile90

	xTrain = xTrain.astype(np.float32)
	# yTrain = yTrain.astype(np.int32)
	xVal = xVal.astype(np.float32)
	# yVal = yVal.astype(np.int32)

	print("Training model...")
	if args.mode=="single-channel":
		xTrain_reshaped = np.reshape(xTrain,(-1,1,magnitude_window,ceil-floor))
		yTrain_repeat = np.repeat(yTrain,args.no_channels)
		netSpec.fit(xTrain_reshaped,yTrain_repeat)
	elif include_userdata:
		netSpec.fit({'sensors':xTrain,'user':udTrain},yTrain)
	else:
		netSpec.fit(xTrain, yTrain)
	gc.collect()

	if not args.no_save_model:
		print("Saving model...")
		modelAndNorm = {'normalization_data':normalization_data,'model':netSpec.get_all_params_values()}
		with open(args.model_filename, 'w') as f:
			pickle.dump(modelAndNorm, f)
	return netSpec 

def load_trained_and_normalize(netSpec, xTrain, xVal):
	print("Loading model...")
	with open(args.model_filename) as f:
		model_norm = pickle.load(f)
	netSpec.load_params_from(model_norm['model'])
	# assert np.equal(modelAndNorm['maximum'], maximum)

	print "Normalizing values "
	xT_freq, xT_bounds = np.histogram(xTrain)
	xV_freq, xV_bounds = np.histogram(xVal)
	print xT_freq/1000
	print xT_bounds/1000
	print xV_freq/1000
	print xV_bounds/1000
	
	# stdev = model_norm['normalization_data']['stdev']
	# mean = model_norm['normalization_data']['mean']
	# print "Normalizing with ", mean, stdev
	# xTrain = (xTrain-mean)*stdev
	# xVal = (xVal-mean)*stdev

	# amin = model_norm['normalization_data']['amin']
	# amax = model_norm['normalization_data']['amax']
	# print "Normalizing with ", amin, amax
	# xTrain = (xTrain-amin)/amax*2 -1
	# xVal = (xVal-amin)/amax*2 -1

	# percentile90 = model_norm['normalization_data']['percentile90']
	# print "Normalizing with percentile90 ", percentile90
	# xTrain = xTrain/percentile90
	# xVal = xVal/percentile90
	if cfg['preprocess']['normalization'] == 'min_max_x255':
		maximum = model_norm['normalization_data']['maximum']
		minimum = model_norm['normalization_data']['minimum']
		print "Normalizing  /maximum*255 ", minimum, maximum
		xTrain = (xTrain-minimum)/(maximum-minimum)*255.0
		xVal = (xVal-minimum)/(maximum-minimum)*255.0

	if cfg['preprocess']['normalization'] == 'log':
		print "Normalizing  log(1+x)*100 "
		xTrain = np.log10(1+xTrain)*100
		xVal = np.log10(1+xVal)*100

	# maximum = model_norm['normalization_data']['maximum']
	# print "Normalizing  log(x)/maximum*2-1 ", maximum
	# xTrain = np.log(xTrain)
	# xVal = np.log(xVal)
	# xTrain = xTrain/maximum*2.0-1.0
	# xVal = xVal/maximum*2.0-1.0

	# mean = model_norm['normalization_data']['mean']
	# stdev = model_norm['normalization_data']['stdev']
	# print "Normalizing with mean ", mean, " stdev ", stdev
	# xTrain = np.log(xTrain)
	# xVal = np.log(xVal)
	# xTrain = (xTrain-mean)/stdev
	# xVal = (xVal-mean)/stdev

	return netSpec, xTrain, xVal

def predict(netSpec, xVal):
	if args.mode=="single-channel":
		pp0 = netSpec.predict_proba(xVal[:,[0]])
		pp1 = netSpec.predict_proba(xVal[:,[1]])
		pp2 = netSpec.predict_proba(xVal[:,[2]])
		pp3 = netSpec.predict_proba(xVal[:,[3]])
		pp = (pp0+pp1+pp2+pp3)/args.no_channels
		return np.argmax(pp,axis=1)
	elif args.shift==3 and args.no_training:
		pp0 = netSpec.predict_proba(xVal[:,:,2:])
		pp1 = netSpec.predict_proba(xVal[:,:,1:-1])
		pp2 = netSpec.predict_proba(xVal[:,:,0:-2])
		pp = (pp0+pp1+pp2)/4
		return np.argmax(pp,axis=1)
	elif args.shift==5 and args.no_training:
		pp0 = netSpec.predict_proba(xVal[:,:,4:])
		pp1 = netSpec.predict_proba(xVal[:,:,3:-1])
		pp2 = netSpec.predict_proba(xVal[:,:,2:-2])
		pp3 = netSpec.predict_proba(xVal[:,:,1:-3])
		pp4 = netSpec.predict_proba(xVal[:,:,0:-4])
		pp = (pp0+pp1+pp2+pp3+pp4)/4
		return np.argmax(pp,axis=1)
	else:
		return netSpec.predict(xVal)

def radicalize(probs,log_decr=1.0):
	print "probs.shape", probs.shape
	radical = np.zeros(probs.shape,dtype=np.float32)
	for k in range(probs.shape[0]):
		for i in range(probs.shape[1]):
			for j in range(i,probs.shape[1]):
				if i!=j:
					if probs[k][i]<probs[k][j]:
						delta = 10 ** (math.log10(probs[k][i])-log_decr)
						radical[k][i] -= delta
						radical[k][j] += delta
					else:
						delta = 10 ** (math.log10(probs[k][j])-log_decr)
						radical[k][j] -= delta
						radical[k][i] += delta
	return probs+radical

def test():
	if cfg['evaluation']['online_training']:
		print("Start evaluation and online training...")
		print("offline_validation...")
		prediction = predict(netSpec, xVal)
		probabilities = netSpec.predict_proba(xVal)
		print("Performance_on_relevant_data")
		result = yVal==prediction
		faults = yVal!=prediction
		acc_val = float(np.sum(result))/float(len(result))
		print "Accuracy_validation: ", acc_val
		print "Error_rate_(%): ", 100*(1-acc_val)
		relTrain = yTrain != label_values.noise
		relVal = yVal != label_values.noise
		print 'Ratio_validation_relevant_data:', float(np.count_nonzero(relVal)) / (np.count_nonzero(relVal) + np.count_nonzero(relTrain))
		rresult = yVal[relVal]==prediction[relVal]
		acc_val_relevant = float(np.sum(rresult))/float(len(rresult))
		print "Accuracy_for_relevant_data: ", acc_val_relevant
		print "Error_rate_for_relevant_data_(%): ", 100*(1-acc_val_relevant)

		prediction = np.zeros((xVal.shape[0]),dtype=np.int32)
		probabilities = np.zeros((xVal.shape[0],2),dtype=np.float32)
		batch_size = 128

		print "xVal.shape[0]", xVal.shape[0]
		for i in range(0,xVal.shape[0]-batch_size,batch_size):
			fragment_xVal = xVal[i:i+batch_size]
			fragment_prediction = predict(netSpec, fragment_xVal)
			prediction[i:i+batch_size] = fragment_prediction
			fragment_probabilities = netSpec.predict_proba(fragment_xVal)
			probabilities[i:i+batch_size] = fragment_probabilities
			new_fragment_probabilities = radicalize(fragment_probabilities)
			print "fragment_xVal.shape", fragment_xVal.shape
			print "new_fragment_probabilities", new_fragment_probabilities

			netSpec.partial_fit(fragment_xVal,new_fragment_probabilities)
	else:	
		print("Validating...")
		if include_userdata:
			prediction = predict(netSpec, {'sensors':xVal,'user':udVal})
			probabilities = netSpec.predict_proba({'sensors':xVal,'user':udVal})
			print "probabilities.shape", probabilities.shape
		else:
			prediction = predict(netSpec, xVal)
			probabilities = netSpec.predict_proba(xVal)
			print "probabilities.shape", probabilities.shape

	print("Showing last 30 test samples..")
	print("Predictions:")
	print(prediction[-30:])
	print("Ground Truth:")
	print(yVal[-30:])
	print("Performance on relevant data")
	result = yVal==prediction
	faults = yVal!=prediction
	acc_val = float(np.sum(result))/float(len(result))
	print "Accuracy validation: ", acc_val
	print "Error rate (%): ", 100*(1-acc_val)
	#print np.nonzero(faults)
	
	print "yVal", yVal
	
	if args.plot_prob_dist:
		rrprobs = probabilities[relVal]
		rrprobs_idx = prediction[relVal]
		rrprobs = rrprobs[np.arange(rrprobs_idx.size),rrprobs_idx]
		rrprobs_correct = rrprobs[rresult] 
		rrprobs_wrong = rrprobs[np.invert(rresult)]

		numBins = 40
		p1 = plt.hist(rrprobs_correct,numBins,color='green',alpha=0.5, label="Correct samples")
		p2 = plt.hist(rrprobs_wrong,numBins,color='red',alpha=0.5, label="Wrong samples")
		max_bin_size = max(max(p1[0]),max(p2[0]))
		plt.plot((np.median(rrprobs_correct), np.median(rrprobs_correct)),(0, max_bin_size), 'g-', label="Median prob for correct samples")
		plt.plot((np.median(rrprobs_wrong), np.median(rrprobs_wrong)),(0, max_bin_size), 'r-', label="Median prob for false samples")
		plt.title("Distribution of predicted probabilities")
		plt.legend(loc='upper center', numpoints=1, bbox_to_anchor=(0.5,-0.05), ncol=2, fancybox=True, shadow=True)
		dest_str = ""
		for session in args.include_session:
			dest_str = dest_str+'_'+session
		plt.savefig('dist_proba'+dest_str+'.png', bbox_inches='tight') 
		plt.show()

	# selVal = aVal['saturated']
	# tresult = yVal[selVal]==prediction[selVal]
	# print "Ratio selection:", float(np.count_nonzero(selVal))/len(xVal)
	# acc_val_sel = float(np.sum(tresult))/float(len(tresult)+0.0001)
	# print "Accuracy for selection", acc_val_sel
	# print "Error rate for selection val data (%): ", 100*(1-acc_val_sel)


	from sklearn.metrics import confusion_matrix
	cm =  confusion_matrix(yVal,prediction)
	print cm
	
	from sklearn.metrics import roc_auc_score
	print roc_auc_score(yVal, probabilities[:,1])



data_path = args.data_path


files_per_hour = 6


#is_train_index = get_train_val_split(size)


if args.no_preprocessing:
	load_preprocessed()
else:
	preprocess()

model_training = None
model_evaluation = None
print "Building models ..."
if include_userdata:
	import convnets.multi_user_models as cnmu
	model_training = getattr(cnmu, cfg['training']['model'])
	print "Model name for the training phase: ", cfg['training']['model']
	model_evaluation = getattr(cnmu, cfg['evaluation']['model'])
	print "Model name for the evaluation phase: ", cfg['evaluation']['model']
else:
	import convnets.models as cn
	model_training = getattr(cn, cfg['training']['model'])
	print "Model name for the training phase: ", cfg['training']['model']
	model_evaluation = getattr(cn, cfg['evaluation']['model'])
	print "Model name for the evaluation phase: ", cfg['evaluation']['model']

if args.mode=="single-channel":
	no_channels = 1
else:
	no_channels = args.no_channels

if args.no_training:
	netSpec = model_evaluation(no_channels,magnitude_window,ceil-floor)
else:
	netSpec = model_training(no_channels,magnitude_window,ceil-floor)	

if args.no_training:
	netSpec, xTrain, xVal = load_trained_and_normalize(netSpec, xTrain, xVal)
else:
	netSpec = normalize_and_train(netSpec)

if args.chosen_validation_ratio != 0:
	test()
