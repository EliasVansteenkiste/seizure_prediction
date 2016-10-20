#!/home/robo_external/.local/bin python
# -*- coding: utf-8 -*-

import global_vars as g
g.init()
import sys
import os
import time
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec

from tools.utils import calcFFT, rolling_window_ext
from tools.analyze import read_data_1h, read_data
from fractions import gcd

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
import psutil
import scipy

from sklearn.base import clone
from datetime import datetime

import datasets

#https://docs.python.org/2/library/argparse.html
parser = argparse.ArgumentParser(description='Preprocess/Train/Validate all data.')
parser.add_argument('--data-path', dest='data_path', action='store', default="/home/eavsteen/seizure_detection/data",
                   help='top level path of data (default: /home/eavsteen/seizure_detection/data)')
parser.add_argument('--model-path', dest='model_path', action='store',
                   default="/home/eavsteen/seizure_detection/vault",
                   help='save/read the model parameters to/from the filename given (default: /home/eavsteen/seizure_detection/vault)')
parser.add_argument('--config-filename', dest='config_filename', action='store',
                   default="config.yml",
                   help='read the configuration parameters from the filename given (default: config.yml)')
parser.add_argument('--no-preprocessing', dest='no_preprocessing', action='store_true', default=False,
                   help='skip preprocessing. load preprocessed data from file (default: false)')
parser.add_argument('--save-preprocessed', dest='save_preprocessed', action='store_true', default=False,
                   help='save the data after preprocessing. saves some time (default: false)')
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
					help='The number of channels that will be used for training and inference (default: 16)')
parser.add_argument('--target-gpu',dest='target_gpu', action='store', default="gpu0",
                   help='target gpu')
parser.add_argument('--mode', dest='mode', action='store', default="None",
                   help='single-channel/dual-channel/none (default: none)')
parser.add_argument('--patients', dest='patients', nargs='+', default=['patient0'],
					help='the target patients')
parser.add_argument('--no-predict-test', dest='no_predict_test', action='store_true', default=False,
                   help='do not calculate the predictions for the test set. (default: false)')


g.args = parser.parse_args()
args = g.args

print "Command line arguments:", args
print "Git reference: ",
os.system("git show-ref HEAD")
print "Timestamp:", datetime.now()
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
	g.cfg = yaml.load(ymlfile)
cfg = g.cfg

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

global patients
patients = dict()

print args.patients
print type(args.patients)
for i in range(len(args.patients)):
	patient = args.patients[i]
	words = patient.split('_')
	user = words[0]
	channels = np.empty((args.no_channels),dtype=np.int32)
	for ch in range(args.no_channels):
		channels[ch] = int(words[ch+1])
	patients[user]=channels

for dataset in datasets.all:
	if dataset.user in patients.keys():
		dataset.enabled = True
	else:
		dataset.enabled = False


def read_train_data(dataset,k_normal_val,k_normal_train,k_seizure_val,k_seizure_train):
	global train_counter_seizure
	global val_counter_seizure
	global train_counter_normal
	global val_counter_normal
	global patients

	print "read data and preprocess (fft and slicing)"
	channels = patients[dataset.user]
	print "read in channels", channels
	
	path = data_path+'/'+dataset.set_name+'/'+dataset.base_name
	print path

	# read in normal 
	is_train_index = get_train_val_split(k_normal_train,k_normal_val)
	no_normal = k_normal_val + k_normal_train
	for i in xrange(no_normal):
		print "normal i", i
		sys.stdout.flush()
		data_1h = read_data_1h(path,'_0.mat',i*6+1)
		ch_arrays = []
		for ch in channels:
			ch_arrays.append(calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil])
		magnitude = np.stack(ch_arrays, axis=0)
		if is_train_index[i]:
			g.magnitudes_normal_train[train_counter_normal] = magnitude
			train_counter_normal += 1
		else:
			g.magnitudes_normal_val[val_counter_normal] = magnitude
			val_counter_normal += 1

	# read in seizure 
	is_train_index = get_train_val_split(k_seizure_train,k_seizure_val)
	no_seizure = k_seizure_val + k_seizure_train
	for i in xrange(no_seizure):
		print "seizure i", i
		sys.stdout.flush()
		data_1h = read_data_1h(path,'_1.mat',i*6+1)
		ch_arrays = []
		for ch in channels:
			ch_arrays.append(calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil])
		magnitude = np.stack(ch_arrays, axis=0)
		if is_train_index[i]:
			g.magnitudes_seizure_train[train_counter_seizure] = magnitude
			train_counter_seizure += 1
		else:
			g.magnitudes_seizure_val[val_counter_seizure] = magnitude
			val_counter_seizure += 1
	
	print "Done reading in", no_normal, "no seizure hours and", no_seizure, "seizure hours"

def read_test_data(dataset,start,stop):
	global magnitudes_test
	global test_counter

	print "read data and preprocess (fft and slicing)"
	channels = patients[dataset.user]
	print "read in channels", channels
	
	path = data_path+'/'+dataset.set_name+'/'+dataset.base_name
	print path

	# read in normal 
	for i in xrange(start,stop):
		#print "test i", i
		sys.stdout.flush()
		data = read_data(path,'.mat',i+1)
		ch_arrays = []
		for ch in channels:
			ch_arrays.append(calcFFT(data[:,ch],fft_width,overlap)[:,floor:ceil])
		magnitude = np.stack(ch_arrays, axis=0)
		magnitudes_test[test_counter] = magnitude
		test_counter += 1


	print "Done reading in", stop-start, "test snippets of 10min."

def get_train_val_split(train_no,val_no):
	no = train_no + val_no
	if args.fixed_seed:
		random.seed(0) 
		np.random.seed(0)
	all_indices = np.arange(no)
	if args.shuffle_before_split:
		np.random.shuffle(all_indices)
	train_file_indices = all_indices[:train_no]
	is_train_index = np.zeros(no, dtype=np.bool)
	is_train_index[train_file_indices] = True
	# val_no = no - train_no
	# val_indices = all_indices[train_no:]
	return is_train_index

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
	global train_counter_seizure
	global val_counter_seizure
	global train_counter_normal
	global val_counter_normal
	global userdata
	global labels
	global analysis_datas

	print("Loading and preprocessing data...")

	no_normal_train = 0
	no_normal_val = 0
	no_seizure_train = 0
	no_seizure_val = 0

	for dataset in datasets.all:
		if dataset.enabled:
			no_normal_val += int(dataset.no_normal * args.debug_sub_ratio * args.chosen_validation_ratio)
			no_normal_train += int(dataset.no_normal * args.debug_sub_ratio * (1-args.chosen_validation_ratio))
			no_seizure_val += int(dataset.no_seizure * args.debug_sub_ratio * args.chosen_validation_ratio)
			no_seizure_train += int(dataset.no_seizure * args.debug_sub_ratio * (1-args.chosen_validation_ratio))
	

	no_normal = no_normal_val + no_normal_train
	no_seizure = no_seizure_val + no_seizure_train

	print "total"
	print no_normal
	print no_seizure
	print "train"
	print no_normal_train
	print no_seizure_train
	print "validation"
	print no_normal_val
	print no_seizure_val
	
	test = read_data_1h(data_path+'/train_1/1_','_0.mat',1)
	test_magnitude = calcFFT(test[:,0],fft_width,overlap)[:,floor:ceil]
	print "test_magnitude.shape", test_magnitude.shape
	stft_steps = test_magnitude.shape[0]


	print no_seizure_train
	print no_seizure-no_seizure_train
	print no_normal_train
	print no_normal-no_normal_train

	g.magnitudes_seizure_train = np.zeros((no_seizure_train,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	g.magnitudes_seizure_val = np.zeros((no_seizure_val,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	g.magnitudes_normal_train = np.zeros((no_normal_train,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	g.magnitudes_normal_val = np.zeros((no_normal_val,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	

	# analysis_datas = np.zeros(size, dtype=analysis_data_type)

	global train_counter_seizure
	global val_counter_seizure
	train_counter_seizure = 0
	val_counter_seizure = 0

	global train_counter_normal
	global val_counter_normal
	train_counter_normal = 0
	val_counter_normal = 0

	no_dss = 0
	for dataset in datasets.all:
		if dataset.enabled:
			no_dss += 1

	for dataset in datasets.all:
		if dataset.enabled and dataset.trainset:
			print "Read in dataset from %s ..."%(dataset.set_name)
			print "Processing data ..."
			k_normal_val = int(dataset.no_normal * args.debug_sub_ratio * args.chosen_validation_ratio)
			k_normal_train = int(dataset.no_normal * args.debug_sub_ratio * (1-args.chosen_validation_ratio))
			k_seizure_val = int(dataset.no_seizure * args.debug_sub_ratio * args.chosen_validation_ratio)
			k_seizure_train = int(dataset.no_seizure * args.debug_sub_ratio * (1-args.chosen_validation_ratio))
			read_train_data(dataset,k_normal_val,k_normal_train,k_seizure_val,k_seizure_train)
			print 'train_counter_seizure', train_counter_seizure, 'val_counter_seizure', val_counter_seizure
			print 'train_counter_normal', train_counter_normal, 'val_counter_normal', val_counter_normal

	process = psutil.Process(os.getpid())
	print("Memory usage (GB): "+str(process.memory_info().rss/1e9))

	print 'train_counter_seizure', train_counter_seizure, 'val_counter_seizure', val_counter_seizure
	print 'train_counter_normal', train_counter_normal, 'val_counter_normal', val_counter_normal

	print "percentiles:"
	for p in range(0,101,10):
		print p, np.percentile(g.magnitudes_normal_train, p), np.percentile(g.magnitudes_normal_val, p)

	#Construct training vector
	train_multiplier = 1
	gcd_ = gcd(no_normal,no_seizure)
	samplesph_normal = train_multiplier * no_seizure / gcd_
	samplesph_seizure = train_multiplier * no_normal / gcd_
	size_train = no_normal_train * samplesph_normal + no_seizure_train * samplesph_seizure
	
	magnitudes = np.hstack((np.arange(size_train/2),np.arange(size_train/2)))
	labels = np.hstack((np.zeros(size_train/2),np.ones(size_train/2)))
	np.random.shuffle(labels)

	yTrain = labels.astype(np.int32)
	xTrain = magnitudes.astype(np.float32)

	print("Histogram:")
	print np.bincount(yTrain)

	print "yTrain.shape", yTrain.shape
	print "xTrain.shape", xTrain.shape

	#Construct validation vector
	val_mult = 3
	gcd_ = gcd(no_normal_val,no_seizure_val)
	samplesph_normal = val_mult * no_seizure_val / gcd_
	samplesph_seizure = val_mult * no_normal_val / gcd_
	size_val = no_normal_val * samplesph_normal + no_seizure_val * samplesph_seizure
	
	magnitudes = np.hstack((np.arange(size_val/2)%no_normal_val,np.arange(size_val/2)%no_seizure_val))
	labels = np.hstack((np.zeros(size_val/2),np.ones(size_val/2)))	

	yVal = labels.astype(np.int32)
	xVal = magnitudes.astype(np.float32)

	print("Histogram:")
	print np.bincount(yVal)

	print "yTrain.shape", yTrain.shape
	print "xTrain.shape", xVal.shape

	print "xVal.shape", xVal.shape
	print "yVal.shape", yVal.shape
	xVal = np.vstack((xVal,yVal))
	xVal = np.swapaxes(xVal,0,1)

	size = size_val +size_train
	no_val = int(math.floor(args.chosen_validation_ratio * size))
	no_train = size-no_val
	assert no_train + no_val == size
	print 'Ratio validation:', no_val/float(size)
	if abs(no_val/float(size) - args.chosen_validation_ratio) > 0.02:
		print "WARNING: validation ratio (%g) differs from expected value (%g)"%(no_val/float(size), args.chosen_validation_ratio)
	

	del magnitudes
	gc.collect()


	print 'xTrain.shape', xTrain.shape
	print 'yTrain.shape', yTrain.shape
	print 'xVal.shape', xVal.shape
	print 'yVal.shape', yVal.shape
	assert xTrain.shape[0] == yTrain.shape[0]
	assert xVal.shape[0] == yVal.shape[0]

	if args.save_preprocessed:
		print("Saving preprocessed data...")
		data = {
			'magnitudes_seizure_val': g.magnitudes_seizure_val,
			'magnitudes_seizure_train': g.magnitudes_seizure_train,
			'magnitudes_normal_val': g.magnitudes_normal_val,
			'magnitudes_normal_train': g.magnitudes_normal_train,
			'xTrain':xTrain, 
			#'udTrain':udTrain, 
			#'aTrain':aTrain, 
			'yTrain':yTrain, 
			'xVal':xVal,
			#'udVal':udVal, 
			'yVal':yVal,
			}
		hkl.dump(data, 'preprocessedData.hkl',compression="lzf")

def preprocess_test_data():
	global magnitudes_test
	global test_counter

	print("Loading and preprocessing data...")

	no_files = 0

	for dataset in datasets.all:
		if dataset.enabled and not dataset.trainset:
			no_files += int(dataset.no_files * args.debug_sub_ratio)

	print "no_files", no_files
	
	test = read_data(data_path+'/test_1/1_','.mat',1)
	test_magnitude = calcFFT(test[:,0],fft_width,overlap)[:,floor:ceil]
	print "test_magnitude.shape", test_magnitude.shape
	stft_steps = test_magnitude.shape[0]

	magnitudes_test = np.zeros((no_files,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	print magnitudes_test.shape
	test_counter = 0


	for dataset in datasets.all:
		if dataset.enabled and not dataset.trainset:
			print "Read in dataset from %s ..."%(dataset.set_name)
			nf = int(dataset.no_files * args.debug_sub_ratio)
			read_test_data(dataset,0,nf)

	process = psutil.Process(os.getpid())
	print("Memory usage (GB): "+str(process.memory_info().rss/1e9))



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
	g.magnitudes_seizure_val = data['magnitudes_seizure_val']
	g.magnitudes_seizure_train = data['magnitudes_seizure_train']
	g.magnitudes_normal_val = data['magnitudes_normal_val']
	g.magnitudes_normal_train = data['magnitudes_normal_train']
	#aVal = data['aVal']
	# if include_userdata:
	# 	udTrain = data['udTrain']
	# 	udVal = data['udVal']	

#@profile
def train(netSpec):
	global xTrain
	global xVal
	global yTrain
	global yVal


	xTrain = xTrain.astype(np.float32)
	xVal = xVal.astype(np.float32)

	yTrain = yTrain.astype(np.int32)
	yVal = yVal.astype(np.int32)

	print("Training model...")
	netSpec.fit(xTrain, yTrain)

	if not args.no_save_model:
		patient_str = '-'.join(args.patients)
		model_filename = patient_str+'_'+cfg['training']['model']+'_'+datetime.now().strftime("%m-%d-%H-%M-%S")+'.pickle'
		print("Saving model...")
		model = {'model':netSpec.get_all_params_values()}
		with open(args.model_path+'/'+model_filename, 'w') as f:
			pickle.dump(model, f)
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

def test():

	print("Validating...")
	print "Changing batch iterator test:"
	from nolearn.lasagne import BatchIterator
	netSpec.batch_iterator_test = BatchIterator(batch_size=128)

	print "Calculating final prediction for the hour long sessions"
	print "magnitudes_normal_val.shape", g.magnitudes_normal_val.shape
	probabilities_hour = []
	probabilities_normal = []
	for mag_hour in g.magnitudes_normal_val:
		patches = rolling_window_ext(mag_hour,(magnitude_window,ceil-floor))
		patches = np.swapaxes(patches,0,2)
		predictions_patches = netSpec.predict_proba(patches[0])
		probabilities_normal.append(predictions_patches)
		prediction_hour = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
		probabilities_hour.append(prediction_hour[1])

	probabilities_seizure = []
	print "magnitudes_seizure_val.shape", g.magnitudes_seizure_val.shape
	for mag_hour in g.magnitudes_seizure_val:
		patches = rolling_window_ext(mag_hour,(magnitude_window,ceil-floor))
		patches = np.swapaxes(patches,0,2)
		predictions_patches = netSpec.predict_proba(patches[0])
		probabilities_seizure.append(predictions_patches)
		prediction_hour = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
		probabilities_hour.append(prediction_hour[1])

	probabilities_normal = np.stack(probabilities_normal)
	probabilities_seizure = np.stack(probabilities_seizure)
	probabilities_normal = np.reshape(probabilities_normal,(-1,2))
	probabilities_seizure = np.reshape(probabilities_seizure,(-1,2))	
	print "probabilities_normal", probabilities_normal.shape
	print "probabilities_seizure", probabilities_seizure.shape	
	yVal = np.hstack((np.zeros(len(probabilities_normal)),np.ones(len(probabilities_seizure))))
	probabilities = np.vstack((probabilities_normal,probabilities_seizure))
	prediction = np.argmax(probabilities,axis=1)
	print("Showing last 30 test samples..")
	print("Predictions:")
	print(probabilities[-30:,1])
	print("Ground Truth:")
	print(yVal[-30:])
	print("Performance on relevant data")
	result = yVal==prediction
	faults = yVal!=prediction
	acc_val = float(np.sum(result))/float(len(result))
	print "Accuracy validation: ", acc_val
	print "Error rate (%): ", 100*(1-acc_val)
	from sklearn.metrics import confusion_matrix
	cm =  confusion_matrix(yVal,prediction)
	print cm
	from sklearn.metrics import roc_auc_score,log_loss
	print probabilities[:,1].shape
	print yVal.shape
	print "roc_auc:", roc_auc_score(yVal, probabilities[:,1])
	print "log_loss:", log_loss(yVal, probabilities[:,1])

	yVal_hour = np.hstack((np.zeros(g.magnitudes_normal_val.shape[0]),np.ones(g.magnitudes_seizure_val.shape[0])))
	print "roc_auc for the hours:", roc_auc_score(yVal_hour, probabilities_hour)
	print "log_loss for the hours:", log_loss(yVal_hour, probabilities_hour)

	print "saving predictions to csv file" 
	patient_str = '-'.join(args.patients)
	csv_filename = 'hours'+patient_str+'_'+cfg['training']['model']+'_'+datetime.now().strftime("%m-%d-%H-%M-%S")+'.csv'
	print csv_filename
	csv=open('./results/'+csv_filename, 'w+')
	for i in range(yVal_hour.shape[0]):
		csv.write(str(yVal_hour[i])+','+str(probabilities_hour[i])+'\n')
	csv.close
	
	predictions_hour = np.round(probabilities_hour)
	result_hour = yVal_hour==predictions_hour
	acc_val_hour = float(np.sum(result_hour))/float(len(result_hour))
	print "Accuracy validation for the hours: ", acc_val_hour

	if not args.no_predict_test:
		print "Calculating the predictions for the test files"
		preprocess_test_data()

		probabilities_test = []
		for mag_test in magnitudes_test:
			patches = rolling_window_ext(mag_test,(magnitude_window,ceil-floor))
			patches = np.swapaxes(patches,0,2)
			predictions_patches = netSpec.predict_proba(patches[0])
			prediction_test = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
			probabilities_test.append(prediction_test[1])

		print "saving predictions to csv file" 
		csv_filename = patient_str+'_'+cfg['training']['model']+'_'+datetime.now().strftime("%m-%d-%H-%M-%S")+'.csv'
		print csv_filename
		csv=open('./results/'+csv_filename, 'w+')
		counter = 0
		for dataset in datasets.all:
			if dataset.enabled and not dataset.trainset:
				for i in range(int(dataset.no_files * args.debug_sub_ratio)):
					filename = dataset.base_name+str(i+1)+'.mat'
					csv.write(filename+','+str(probabilities_test[counter+i])+'\n')
		csv.close



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

import batch_iterators
if args.no_training:
	netSpec = model_evaluation(no_channels,magnitude_window,ceil-floor,batch_iterator_train=batch_iterators.BI_train_bal_complete(16),batch_iterator_test=batch_iterators.BI_test_bal_complete(16))
else:
	netSpec = model_training(no_channels,magnitude_window,ceil-floor,batch_iterator_train=batch_iterators.BI_train_bal_complete(16),batch_iterator_test=batch_iterators.BI_test_bal_complete(16))	

if args.no_training:
	netSpec, xTrain, xVal = load_trained_and_normalize(netSpec, xTrain, xVal)
else:
	netSpec = train(netSpec)

if args.chosen_validation_ratio != 0:
	test()
