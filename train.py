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
from train_split import TrainSplit
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
import operator

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
                   help='fixed random seed. (default: True)')
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
                   help='no prediction of the competition test set. (default: False)')

#Added to deal with Kaggle's fuckup
parser.add_argument('--labels', dest='labels', action='store', default="train_and_test_data_labels_safe.csv",
                   help='labels and safe flags (default: train_and_test_data_labels_safe.csv)')
parser.add_argument('--new-data', dest='new_data', action='store', default="train_and_test_data_labels_new.csv",
                   help='new data csv (default: train_and_test_data_labels_new.csv)')



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
m_window = preprocess_params['magnitude_window']
include_userdata = preprocess_params['include_userdata']

height=fft_width/2
assert ceil-floor <= fft_width / 2
assert ceil <= fft_width / 2

global chs
chs = dict()

print args.patients
print type(args.patients)
for i in range(len(args.patients)):
	patient = args.patients[i]
	words = patient.split('_')
	user = words[0]
	channels = np.empty((args.no_channels),dtype=np.int32)
	for ch in range(args.no_channels):
		channels[ch] = int(words[ch+1])
	chs[user]=channels

for dataset in datasets.all:
	if dataset.user in chs.keys():
		dataset.enabled = True
	else:
		dataset.enabled = False

global legal_seizure_files
legal_seizure_files = dict()
global legal_normal_files
legal_normal_files = dict()

blacklist = []
with open('./'+args.labels) as f:
	for line in f:
	 	words = line.rsplit()[0].split(',')
	 	#only keep the safe files
		if words[2] == '1':
			if words[1] == '0':
				patient = 'patient'+words[0].split('_')[0]
				if patient in legal_normal_files:
					legal_normal_files[patient].append(words[0])
				else:
					legal_normal_files[patient] = []
			else:
				patient = 'patient'+words[0].split('_')[0]
				if patient in legal_seizure_files:
					legal_seizure_files[patient].append(words[0])
				else:
					legal_seizure_files[patient] = []
		else:
			blacklist.append(words[0])

print 'legal_seizure_files'
for patient in legal_seizure_files.keys():
	print patient, ': ', len(legal_seizure_files[patient])
	for dataset in datasets.all:
		if dataset.user == patient and dataset.trainset:
			dataset.no_seizure_files_clean = len(legal_seizure_files[patient])
			print 'dataset.no_seizure_files_clean', dataset.no_seizure_files_clean
			dataset.no_seizure_clean = dataset.no_seizure_files_clean/6
			print 'dataset.no_seizure_clean', dataset.no_seizure_clean

print 'legal_normal_files'
for patient in legal_normal_files.keys():
	print patient, ': ', len(legal_normal_files[patient])
	for dataset in datasets.all:
		if dataset.user == patient and dataset.trainset:
			dataset.no_normal_files_clean = len(legal_normal_files[patient])
			dataset.no_normal_clean = dataset.no_normal_files_clean/6


def read_train_data(dataset,no_normal,no_seizure):
	global counter_seizure
	global counter_normal

	print "read data and preprocess (fft and slicing)"
	channels = chs[dataset.user]
	print "read in channels", channels
	
	path = data_path+'/'+dataset.set_name+'/'+dataset.base_name
	print path

	# read in normal
	counter = 0
	print 'dataset.no_normal', dataset.no_normal
	print 'no_normal', no_normal
	for i in xrange(dataset.no_normal):
		print "normal i", i, 
		sys.stdout.flush()
		filename = dataset.base_name + str(i*6+1) + '_0.mat'
		if filename in legal_normal_files[dataset.user]:
			data_1h = read_data_1h(path,'_0.mat',i*6+1)
			ch_arrays = []
			for ch in channels:
				ch_arrays.append(calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil])
			magnitude = np.stack(ch_arrays, axis=0)
			print 'counter', counter
			g.ms_normal[counter_normal] = magnitude
			counter_normal += 1
			counter += 1
		print counter, no_normal
		if counter >= no_normal:
			break;

	# read in seizure 
	counter = 0
	for i in xrange(dataset.no_seizure):
		print "seizure i", i,
		sys.stdout.flush()
		filename = dataset.base_name + str(i*6+1) + '_1.mat'
		if filename in legal_seizure_files[dataset.user]:
			data_1h = read_data_1h(path,'_1.mat',i*6+1)
			ch_arrays = []
			for ch in channels:
				ch_arrays.append(calcFFT(data_1h[:,ch],fft_width,overlap)[:,floor:ceil])
			magnitude = np.stack(ch_arrays, axis=0)
			g.ms_seizure[counter_seizure] = magnitude
			counter_seizure += 1
			counter += 1
		if counter >= no_seizure:
			break;
	
	print "Done reading in", no_normal, "no seizure hours and", no_seizure, "seizure hours"

# extra code for Kaggle's fuckup
def read_extra_dataset(dataset):
	global extra_counter

	print "read data and preprocess (fft and slicing)"
	channels = chs[dataset.user]
	print "read in channels", channels
	
	path = data_path+'/'+dataset.set_name+'/'+dataset.base_name
	print path

	# read in normal 
	for i in dataset.file_indices_whitelist():
		print "xtra ", dataset.user, i
		sys.stdout.flush()
		data = read_data(path,'.mat',i+1)
		ch_arrays = []
		for ch in channels:
			ch_arrays.append(calcFFT(data[:,ch],fft_width,overlap)[:,floor:ceil])
		magnitude = np.stack(ch_arrays, axis=0)
		g.ms_xtra_seizure[extra_counter] = magnitude
		extra_counter += 1

	print "Done reading in", len(dataset.file_indices_whitelist()), "test snippets of 10min."

def set_white_lists():
	
	with open('./'+args.new_data) as f:
		print f.readline() # read the headers
		for line in f:
		 	words = line.rsplit()[0].split(',')
		 	#only keep the safe files
			if words[2] == '1':
				if words[1] == '1':
					patient = 'patient'+words[0].split('_')[0]
					for ds in datasets.new_datasets:
						if ds.user == patient:
							ds.whitelist.append(int(words[0].split('_')[1].split('.')[0]))
				else:
					print 'Warning: CSV file should only contain seizure files.'
					print words
			else:
				print 'Warning: CSV file should only contain safe files.'
				print words

def allocate_array():
	global extra_counter

	n_samples = 0

	extra_counter = 0
	for dataset in datasets.new_datasets:
		if dataset.user in chs.keys():
			n_samples += len(dataset.file_indices_whitelist())

	test = read_data(data_path+'/test_1/1_','.mat',1)
	test_magnitude = calcFFT(test[:,0],fft_width,overlap)[:,floor:ceil]
	stft_steps = test_magnitude.shape[0]

	g.ms_xtra_seizure = np.zeros((n_samples,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)

def read_extra_data():

	set_white_lists()
	allocate_array()


	for dataset in datasets.new_datasets:
		if dataset.user in chs.keys():
			print dataset.user, 'in', chs.keys()
			read_extra_dataset(dataset)

def read_test_data(dataset,start,stop):
	global ms_test
	global test_counter

	print "read data and preprocess (fft and slicing)"
	channels = chs[dataset.user]
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
		ms_test[test_counter] = magnitude
		test_counter += 1

	print "Done reading in", stop-start, "test snippets of 10min."

def get_train_val_split(train_no,val_no,fold=1):
	no = train_no + val_no
	is_train_index = np.zeros(no, dtype=np.bool)
	is_train_index[:train_no] = True
	# val_no = no - train_no
	# val_indices = all_indices[train_no:]
	return is_train_index

def normalize():
	global maximum
	global minimum

	print "percentiles:"
	for p in range(0,101,10):
		print p, np.percentile(g.ms_normal, p), np.percentile(g.ms_seizure, p)

	max1 = np.amax(g.ms_seizure)
	max2 = np.amax(g.ms_normal)
	max3 = np.amax(g.ms_xtra_seizure)
	maximum = max([max1,max2,max3])

	min1 = np.amin(g.ms_seizure)
	min2 = np.amin(g.ms_normal)
	min3 = np.amin(g.ms_xtra_seizure)
	minimum = max([min1,min2,min3])

	if cfg['preprocess']['normalization'] == 'div_max_x255':
		print "Normalizing/maximum*255 ", maximum
		g.ms_seizure = g.ms_seizure/maximum*255.0
		g.ms_normal = g.ms_normal/maximum*255.0
		g.ms_xtra_seizure = g.ms_xtra_seizure/maximum*255.0


	if cfg['preprocess']['normalization'] == 'min_max_x255':
		print "Normalizing-minimum)/(maximum-minimum)*255 ", maximum, minimum
		g.ms_seizure = (g.ms_seizure-minimum)/(maximum-minimum)*255.0
		g.ms_normal = (g.ms_normal-minimum)/(maximum-minimum)*255.0
		g.ms_xtra_seizure = (g.ms_xtra_seizure-minimum)/(maximum-minimum)*255.0

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

def preprocess():
	global size
	global x
	global counter_seizure
	global counter_normal

	print("Loading and preprocessing data...")


	read_extra_data()

	no_normal = 0
	no_seizure = 0

	for dataset in datasets.all:
		if dataset.enabled and dataset.trainset:
			no_normal += int(dataset.no_normal_clean * args.debug_sub_ratio)
			no_seizure += int(dataset.no_seizure_clean * args.debug_sub_ratio)
	

	print 'total', 'no_normal', no_normal, 'no_seizure', no_seizure
	
	test = read_data_1h(data_path+'/train_1/1_','_0.mat',1)
	test_magnitude = calcFFT(test[:,0],fft_width,overlap)[:,floor:ceil]
	print "test_magnitude.shape", test_magnitude.shape
	stft_steps = test_magnitude.shape[0]

	print no_seizure
	print no_normal

	g.ms_seizure = np.zeros((no_seizure,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	g.ms_normal = np.zeros((no_normal,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	
	counter_seizure = 0
	counter_normal = 0

	no_dss = 0
	for dataset in datasets.all:
		if dataset.enabled and dataset.trainset:
			no_dss += 1

	for dataset in datasets.all:
		if dataset.enabled and dataset.trainset:
			print "Read in dataset from %s ..."%(dataset.set_name)
			print "Processing data ..."
			k_normal = int(dataset.no_normal_clean * args.debug_sub_ratio)
			k_seizure = int(dataset.no_seizure_clean * args.debug_sub_ratio)
			read_train_data(dataset,k_normal,k_seizure)

	process = psutil.Process(os.getpid())
	print("Memory usage (GB): "+str(process.memory_info().rss/1e9))

	normalize()

	#Construct data vector
	x = dict()
	x['normal'] = range(len(g.ms_normal))
	x['seizure'] = range(len(g.ms_seizure))
	x['xtra_seizure'] = range(len(g.ms_xtra_seizure))

	if args.save_preprocessed:
		print("Saving preprocessed data...")
		data = {
			'ms_seizure': g.ms_seizure,
			'ms_normal': g.ms_normal,
			'ms_xtra_seizure': g.ms_xtra_seizure,
			'minimum': minimum,
			'maximum': maximum,
			}
		hkl.dump(data, 'preprocessedData.hkl',compression="lzf")

def apply_normalization(data_in):
	global maximum
	global minimum

	if cfg['preprocess']['normalization'] == 'div_max_x255':
		print "Normalizing/maximum*255 ", maximum
		data_out = data_in/maximum*255.0

	if cfg['preprocess']['normalization'] == 'min_max_x255':
		print "Normalizing-minimum)/(maximum-minimum)*255 ", maximum, minimum
		data_out = (data_in-minimum)/(maximum-minimum)*255.0

	if cfg['preprocess']['normalization'] == 'log':
		print "Normalizing  log(1+x)*100 "
		data_out = np.log10(1+data_in)*100

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

	return data_out

def preprocess_test_data():
	global ms_test
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

	ms_test = np.zeros((no_files,args.no_channels,stft_steps,ceil-floor), dtype=np.float32)
	print ms_test.shape
	test_counter = 0


	for dataset in datasets.all:
		if dataset.enabled and not dataset.trainset:
			print "Read in dataset from %s ..."%(dataset.set_name)
			nf = int(dataset.no_files * args.debug_sub_ratio)
			read_test_data(dataset,0,nf)

	ms_test = apply_normalization(ms_test)

	process = psutil.Process(os.getpid())
	print("Memory usage (GB): "+str(process.memory_info().rss/1e9))

def load_preprocessed():
	global minimum
	global maximum

	#global include_userdata
	print("Loading preprocessed data....")
	data = hkl.load('preprocessedData.hkl')
	g.ms_normal = data['ms_normal']
	g.ms_seizure = data['ms_seizure']
	g.ms_seizure = data['ms_xtra_seizure']
	minimum = data['minimum']
	maximum = data['maximum']

#@profile
def train(netSpec):
	global x
	global maximum
	global minimum

	print("Training model...")
	#Only for passing nolearn's check
	y = {'normal': np.zeros(len(x['normal'])),
		'seizure': np.zeros(len(x['seizure'])),
		'xtra_seizure': np.zeros(len(x['xtra_seizure'])),}

	# The brackets are a hack to avoid nolearn's check 
	netSpec.fit([x], [y]) 

	if not args.no_save_model:
		patient_str = '-'.join(args.patients)
		model_filename = patient_str+'_'+cfg['training']['model']+'_'+datetime.now().strftime("%m-%d-%H-%M-%S")+'.pickle'
		print("Saving model...")
		model = {'model':netSpec.get_all_params_values(), 'minimum':minimum, 'maximum':maximum}
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
	return netSpec.predict(xVal)


def check_magnitudes():
	check(g.ms_seizure)
	check(g.ms_normal)

def check(m_array):
	for m in m_array:
		if np.sum(m)<0.1:
			print 'Warning: found a zero or near zero sum array'

def test(netSpec, fold, no_folds):
	print("Validating...")
	print "Changing batch iterator test:"
	from nolearn.lasagne import BatchIterator
	netSpec.batch_iterator_test = BatchIterator(batch_size=128)

	train_split = TrainSplit(no_folds,fold)
	x_train, x_valid, dummy_0, dummy_1 = train_split([x],None)

	print "Calculating final prediction for the hour long sessions"
	print "ms_normal.shape", g.ms_normal.shape
	probabilities_normal_hour = []
	probabilities_normal = []
	for hour in x_valid['normal']:
		mag_hour = g.ms_normal[hour]
		patches = rolling_window_ext(mag_hour,(m_window,ceil-floor))
		patches = np.swapaxes(patches,0,2)
		predictions_patches = netSpec.predict_proba(patches[0])
		probabilities_normal.append(predictions_patches)
		prediction_hour = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
		probabilities_normal_hour.append(prediction_hour[1])

	probabilities_seizure_hour = []
	probabilities_seizure = []
	print "ms_seizure.shape", g.ms_seizure.shape
	for hour in x_valid['seizure']:
		mag_hour = g.ms_seizure[hour]
		patches = rolling_window_ext(mag_hour,(m_window,ceil-floor))
		patches = np.swapaxes(patches,0,2)
		predictions_patches = netSpec.predict_proba(patches[0])
		probabilities_seizure.append(predictions_patches)
		prediction_hour = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
		probabilities_seizure_hour.append(prediction_hour[1])

	probabilities_xtra_seizure_hour = []
	probabilities_xtra_seizure = []
	print "ms_xtra_seizure.shape", g.ms_xtra_seizure.shape
	for hour in x_valid['xtra_seizure']:
		mag_hour = g.ms_xtra_seizure[hour]
		patches = rolling_window_ext(mag_hour,(m_window,ceil-floor))
		patches = np.swapaxes(patches,0,2)
		predictions_patches = netSpec.predict_proba(patches[0])
		probabilities_xtra_seizure.append(predictions_patches)
		prediction_hour = np.sum(predictions_patches,axis=0)/predictions_patches.shape[0]
		probabilities_xtra_seizure_hour.append(prediction_hour[1])

	probabilities_normal = np.stack(probabilities_normal)
	for p in probabilities_seizure:
		print p.shape
	probabilities_seizure = np.stack(probabilities_seizure)
	probabilities_xtra_seizure = np.stack(probabilities_xtra_seizure)
	probabilities_normal = np.reshape(probabilities_normal,(-1,2))
	probabilities_seizure = np.reshape(probabilities_seizure,(-1,2))
	probabilities_xtra_seizure = np.reshape(probabilities_xtra_seizure,(-1,2))	
	print "probabilities_normal", probabilities_normal.shape
	print "probabilities_seizure", probabilities_seizure.shape	
	print "probabilities_xtra_seizure", probabilities_xtra_seizure.shape
	yVal = np.hstack((np.zeros(len(probabilities_normal)),np.ones(len(probabilities_seizure)),np.ones(len(probabilities_xtra_seizure))))
	probabilities = np.vstack((probabilities_normal,probabilities_seizure,probabilities_xtra_seizure))
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
	roc_auc = roc_auc_score(yVal, probabilities[:,1])
	print "roc_auc:", roc_auc
	print "log_loss:", log_loss(yVal, probabilities[:,1])


	yVal_hour = np.hstack((np.zeros(len(probabilities_normal_hour)),np.ones(len(probabilities_seizure_hour)),np.ones(len(probabilities_xtra_seizure_hour))))
	probabilities_hour = probabilities_normal_hour + probabilities_seizure_hour + probabilities_xtra_seizure_hour		
	roc_auc_hours = roc_auc_score(yVal_hour, probabilities_hour)
	print "roc_auc for the hours:", roc_auc_hours
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
		for mag_test in ms_test:
			patches = rolling_window_ext(mag_test,(m_window,ceil-floor))
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

	return roc_auc, roc_auc_hours

data_path = args.data_path

files_per_hour = 6


if args.no_preprocessing:
	load_preprocessed()
else:
	preprocess()

check_magnitudes()

def train_and_test(fold=0,no_folds=5):
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

	from batch_iterators import BI_new

	if args.no_training:
		netSpec = model_evaluation(no_channels,m_window,ceil-floor,
			train_split = TrainSplit(no_folds,fold),
			batch_iterator_train=BI_new(16),batch_iterator_test=BI_new(128))
		netSpec, xTrain, xVal = load_trained_and_normalize(netSpec, xTrain, xVal)
	else:
		netSpec = model_training(no_channels,m_window,ceil-floor,
			train_split = TrainSplit(no_folds,fold),
			batch_iterator_train=BI_new(16),batch_iterator_test=BI_new(128))
		netSpec = train(netSpec)	

	if args.chosen_validation_ratio != 0:
		return test(netSpec, fold, no_folds)

def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))

no_folds = 5
roc_auc_lst = []
roc_auc_hours_lst = []
for i in range(no_folds):
	roc_auc, roc_auc_hours = train_and_test(i,no_folds)
	roc_auc_lst.append(roc_auc)
	roc_auc_hours_lst.append(roc_auc_hours)

print 'geomean_roc_auc: ', geometric_mean(roc_auc_lst)
print 'geomean_roc_auc_hours: ', geometric_mean(roc_auc_hours_lst)


