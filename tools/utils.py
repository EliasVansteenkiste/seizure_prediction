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

def rolling_window_ext(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

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

