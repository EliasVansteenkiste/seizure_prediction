import hickle as hkl
import numpy as np
import time

from numpy.lib.stride_tricks import as_strided as ast

np.set_printoptions(threshold=800)

def calcFFT(data, window, overlap):
	windows = subsequences(data,window,overlap)
	dft = np.fft.fft(windows*np.hamming(window))

	return np.abs(dft)

def subsequences(a, window, overlap):
    #for 1D arrays
    #print "a", a
    #print "a.shape", a.shape
    shape = ((a.size - window)/(window-overlap) + 1, window)
    #print "shape", shape
    #print "a.strides", a.strides
    strides = list(a.strides * 2)
    #print "strides = list(a.strides * 2) = ", strides
    strides[0] = strides[0]*(window-overlap)
    #print "strides[0] = strides[0]*overlap = ", strides[0]
    #print "np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)"
    #print np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
    #for 2d arrays
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def createBufferWindow(data, labels, sampleDistance, window = 2):
	# Dimension of data F x T
	data = data.T
	
	indx = np.arange(0,data.shape[1])
	samplePoints = int(data.shape[1]/sampleDistance)
	while(samplePoints*sampleDistance+(window-1)>data.shape[1]):
		samplePoints = samplePoints - 1	

	indexRange = np.lib.stride_tricks.as_strided(indx, shape=(window,samplePoints), strides=[data.itemsize,data.itemsize*sampleDistance])	

	return np.column_stack(data.T[indexRange,:]), np.max(labels[indexRange], axis = 0)

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

def spectogram(data,window,overlap):
	#data, action =  read(filename)
	data = data[0:56000]
	windows = subsequences(data,window,overlap)

	dft = np.fft.fft(windows*np.hamming(window))

	magnitude = np.abs(dft)
	angle = np.angle(dft)

	return magnitude, angle

def read(filename):
    lib = hkl.load(filename)
    return lib['data'], lib['id'] 

# print "calcFFT test:"
# for sample in range(2):
#     for i in range(2):
#         for overlap in [64,98]:
#             fname = "/local/.dump/data/session20160507/Elias_L_"+str(sample)+"_ch"+str(i)+".raw"
#             data, action =  read(fname)
#             dft = calcFFT(data,128,overlap)
#             print "sample, i, overlap, dft.shape", sample, i, overlap, dft.shape

# print "calcFFT test:"
# for sample in range(2):
#     for i in range(2):
#         for overlap in [128,192]:
#             fname = "/local/.dump/data/session20160507/Elias_L_"+str(sample)+"_ch"+str(i)+".raw"
#             data, action =  read(fname)
#             dft = calcFFT(data,256,overlap)
#             print "sample, i, overlap, dft.shape", sample, i, overlap, dft.shape

