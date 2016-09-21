import numpy as np
import os
import sys
import random
import csv
import math

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#import cv2

import cPickle as pickle

def shuffle(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis,-1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def processDataSpectrum(data, labels, patchSize = 128, patchDisplacement = 8, testFraction = 0.1):
	# data: n x m x k 
	# n: the recording
	# m: time
	# k: bin

	negative_chance = 0.9
	percentOccurrence = 0.6

	numRecordings = data.shape[0]
	numTimeStep = data.shape[1]
	numFreqBins = data.shape[2]

	numPatches = np.floor((numTimeStep-patchSize)/patchDisplacement)+1
	# Total number of patches we will consider.
	nTotalPatches = numRecordings*numPatches
	# (n_total_patches, 3, patchsize, patchsize)
	xTrain = np.zeros((nTotalPatches, 1, patchSize, numFreqBins), dtype='float32')
	yTrain = np.zeros((nTotalPatches,))

	xVal = np.zeros((nTotalPatches, 1, patchSize, numFreqBins), dtype='float32')
	yVal = np.zeros((nTotalPatches,))

	indexTrain=0
	n_files = 0
	indexVal=0

	
	for recording in range(0,numRecordings):
		for col in range(patchSize/2,numTimeStep-patchSize/2, patchDisplacement):
			croppedData = data[recording,col-patchSize/2:col+patchSize/2,:]
			croppLabel = labels[recording,col-patchSize/2:col+patchSize/2]
			containsAction = False

			# Check the labeled action in this patch. If it contains a label different to 4 there is an action
			if np.sum(croppLabel)/len(croppLabel)>0:
				
				foundAction = croppLabel[croppLabel!=0][0]

				#if len(np.where(croppLabel==foundAction)[0])>percentOccurrence*len(croppLabel):
				containsAction = True
				# Add data to train set
				xTrain[indexTrain,0] = croppedData
				yTrain[indexTrain] = int(foundAction)-1 
				#print indexTrain, int(foundAction)

				indexTrain+=1

				continue

			# We didn't detect an action. Add patch with a small probability to the train set. We don't add all the patches as this will blow up the size of the train set with negative examples.
			if not containsAction:
				rand = random.uniform(0,1)
				if rand < negative_chance:
					xTrain[indexTrain,0] = croppedData
					yTrain[indexTrain] = 0

					indexTrain+=1

		n_files = n_files+1
		#sys.stdout.write("\r{0}".format(n_files));
		#sys.stdout.flush()

	xTrain *= float(255.0/xTrain.max())
	
	xTrain = xTrain[0:indexTrain,0,:,:]
	xShape = xTrain.shape
	xTrain = xTrain.reshape((xShape[0],1,xShape[1],xShape[2])).astype(np.float32)
	yTrain = yTrain[0:indexTrain].astype(np.int32)

	nrTestDataPoints = int(len(xTrain)*testFraction)

	xTrain, xVal = xTrain[:-nrTestDataPoints], xTrain[-nrTestDataPoints:]
	yTrain, yVal = yTrain[:-nrTestDataPoints], yTrain[-nrTestDataPoints:]

	return [xTrain, yTrain, xVal, yVal]
