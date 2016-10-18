import theano
import theano.tensor as T
try:
    from theano.tensor.extra_ops import searchsorted
except:
    raise "FUCKING UPDATE YOUR THEANO TO VERSION ZERO POINT FUCKING NINE"
import numpy as np
import pandas as pd
import lasagne
from lasagne.layers import get_output



"""
note: we are following the sklearn api for metrics/loss functions,
where the first arg for a function is y true, and second value is
y predicted. this is the opposite of the theano functions, so just
keep in mind.
"""

#copy existing code and place in tmetrics namespace
multiclass_hinge_loss = lambda yt, yp: lasagne.objectives.multiclass_hinge_loss(yp, yt)
squared_error = lambda yt, yp: lasagne.objectives.squared_error(yp, yt)
binary_accuracy = lambda yt, yp: lasagne.objectives.binary_accuracy(yp, yt)
categorical_accuracy = lambda yt, yp: lasagne.objectives.categorical_accuracy(yp, yt)

def binary_crossentropy(y_true, y_predicted):
    """
    wrapper of theano.tensor.nnet.binary_crossentropy
    args reversed to match tmetrics api
    """
    return T.nnet.binary_crossentropy(y_predicted, y_true)

def categorical_crossentropy(y_true, y_predicted):
    """
    wrapper of theano.tensor.nnet.categorical_crossentropy
    args reversed to match tmetrics api
    """
    return T.nnet.binary_crossentropy(y_predicted, y_true)

def binary_hinge_loss(y_true, y_predicted, binary=True, delta=1):
    """
    wrapper of lasagne.objectives.binary_hinge_loss
    args reversed to match tmetrics api
    """
    return lasagne.objectives.binary_hinge_loss(y_predicted, y_true, binary, delta)




def brier_score_loss(y_true, y_predicted, sample_weight=None):
    """
    port of sklearn.metrics.brier_score_loss
    works for 2D binary data as well, e.g.
    y_true:          [[0, 1, 0],
                     [1, 0, 0]]
    y_predicted:    [[.1, .9, .3],
                     [.4, .7, .2]]
    y_true: tensor, y true (binary)
    y_predicted: tensor, y predicted (float between 0 and 1)  
    sample_weight: tensor or None (standard mean)
    assumptions: 
     -binary ground truth values ({0, 1}); no pos_label
        training wheels like sklearn or figuring out how to 
        run this on text labels. 
     -probabilities are floats between 0-1
     -sample_weight broadcasts to ((y_true - y_predicted) ** 2)
    """
    scores = ((y_true - y_predicted) ** 2)
    if sample_weight is not None: 
        scores = scores * sample_weight
    return scores.mean()

def hamming_loss(y_true, y_predicted):
    """
    note - works on n-dim arrays, means across the final axis
    note - we round predicted because float probabilities would not work
    """
    return T.neq(y_true, T.round(y_predicted)).astype(theano.config.floatX).mean(axis=-1)

def jaccard_similarity(y_true, y_predicted):
    """
    y_true: tensor ({1, 0})
    y_predicted: tensor ({1, 0})
    note - we round predicted because float probabilities would not work
    """
    y_predicted = T.round(y_predicted).astype(theano.config.floatX)
    either_nonzero = T.or_(T.neq(y_true, 0), T.neq(y_predicted, 0))
    return T.and_(T.neq(y_true, y_predicted), either_nonzero).sum(axis=-1, dtype=theano.config.floatX) / either_nonzero.sum(axis=-1, dtype=theano.config.floatX)
        
        
def _nbool_correspond_all(u, v):
    """
    port of scipy.spatial.distance._nbool_correspond_all
    with dtype assumed to be integer/float (no bool in theano)
    sums are on last axis
    """
    not_u = 1.0 - u
    not_v = 1.0 - v
    nff = (not_u * not_v).sum(axis=-1, dtype=theano.config.floatX)
    nft = (not_u * v).sum(axis=-1, dtype=theano.config.floatX)
    ntf = (u * not_v).sum(axis=-1, dtype=theano.config.floatX)
    ntt = (u * v).sum(axis=-1, dtype=theano.config.floatX)
    return (nff, nft, ntf, ntt)

def kulsinski_similarity(y_true, y_predicted):
    y_predicted = T.round(y_predicted)
    nff, nft, ntf, ntt = _nbool_correspond_all(y_true, y_predicted)
    n = y_true.shape[0].astype('float32')
    return (ntf + nft - ntt + n) / (ntf + nft + n)
     
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    reference implementation: numpy.trapz 
    ---------
    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        If `x` is None, then spacing between all `y` elements is `dx`.
    dx : scalar, optional
        If `x` is None, spacing given by `dx` is assumed. Default is 1.
    axis : int, optional
        Specify the axis.
    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule.
    See Also
    --------
    sum, cumsum
    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.
    References
    ----------
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule
    .. [2] Illustration image:
           http://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png
    Examples
    --------
    >>> np.trapz([1,2,3])
    4.0
    >>> np.trapz([1,2,3], x=[4,6,8])
    8.0
    >>> np.trapz([1,2,3], dx=2)
    8.0
    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> np.trapz(a, axis=1)
    array([ 2.,  8.])
    """
    if x is None:
        d = dx
    else:
        if x.ndim == 1:
            d = T.extra_ops.diff(x)
            # reshape to correct shape
            shape = T.ones(y.ndim, dtype='int8')
            shape = T.set_subtensor(shape[axis], d.shape[0])
            d = d.reshape(shape)
        else:
            d = T.extra_ops.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    return (d * (y[slice1] + y[slice2]) / 2.0).sum(axis)

def auc(x, y):
    return abs(trapz(y, x))

#def roc_curve(y_true, y_predicted):
#    fps, tps, thresholds = _binary_clf_curve(y_true, y_predicted)
#    fpr = fps.astype('float32') / fps[-1]
#    tpr = tps.astype('float32') / tps[-1]
#    return fpr, tpr, thresholds
#
#def roc_auc_score(y_true, y_predicted):
#    fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
#    return auc(fpr, tpr)

def _last_axis_binary_clf_curve(y_true, y_predicted):
    """
    returns y_predicted.shape[-2] binary clf curves calculated axis[-1]-wise
    this is a numpy implementation
    """
    assert y_true.shape == y_predicted.shape
    axis = -1
    sort_idx = list(np.ogrid[[slice(x) for x in y_predicted.shape]])
    sort_idx[axis] = y_predicted.argsort(axis=axis).astype('int8')
    reverse = [slice(None)] * y_predicted.ndim
    reverse[axis] = slice(None, None, -1)
    sorted_y_predicted = y_predicted[sort_idx][reverse]
    sorted_y_true = y_true[sort_idx][reverse]


    tps = sorted_y_true.cumsum(axis=axis)
    count = (np.ones(y_predicted.shape) * np.arange(y_predicted.shape[-1]))
    fps = 1 + count - tps
    threshold_values = sorted_y_predicted

    return fps, tps, threshold_values

def last_axis_roc_curve(y_true, y_predicted):
    "numpy implementation"
    fps, tps, thresholds = _last_axis_binary_clf_curve(y_true, y_predicted)
    i = [slice(None)] * fps.ndim
    i[-1] = -1
    fpr = fps.astype('float32') / np.expand_dims(fps[i], axis=-1)
    tpr = tps.astype('float32') / np.expand_dims(tps[i], axis=-1)
    #tpr = tps.astype('float32') / tps[i][:, np.newaxis]
    return fpr, tpr, thresholds

def last_axis_roc_auc_scores(y_true, y_predicted):
    fpr, tpr, _ = last_axis_roc_curve(y_true, y_predicted)
    return np.trapz(tpr, fpr)

def _vector_clf_curve(y_true, y_predicted):
    """
    sklearn.metrics._binary_clf_curve port
    y_true: tensor (vector): y true
    y_predicted: tensor (vector): y predicted
    returns: fps, tps, threshold_values
    fps: tensor (vector): false positivies
    tps: tensor (vector): true positives
    threshold_values: tensor (vector): value of y predicted at each threshold 
        along the curve
    restrictions: 
        -not numpy compatible
        -only works with two vectors (not matrix or tensor)
    """
    assert y_true.ndim == y_predicted.ndim == 1

    desc_score_indices = y_predicted.argsort()[::-1].astype('int8')
    sorted_y_predicted = y_predicted[desc_score_indices]
    sorted_y_true = y_true[desc_score_indices]

    distinct_value_indices = (1-T.isclose(T.extra_ops.diff(sorted_y_predicted), 0)).nonzero()[0]
    curve_cap = T.extra_ops.repeat(sorted_y_predicted.size - 1, 1)
    threshold_indices = T.concatenate([distinct_value_indices, curve_cap]).astype('int8')

    tps = T.extra_ops.cumsum(sorted_y_true[threshold_indices])
    fps = 1 + threshold_indices - tps
    threshold_values = sorted_y_predicted[threshold_indices]

    return fps, tps, threshold_values
 
def _matrix_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 2
    row_i = T.arange(y_true.shape[0], dtype='int8').dimshuffle(0, 'x')
    col_i = y_predicted.argsort().astype('int8')
    reverse = [slice(None), slice(None, None, -1)]
    y_true = y_true[row_i, col_i][reverse]
    y_predicted = y_predicted[row_i, col_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1], dtype='int8')
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _tensor3_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 3
    x_i = T.arange(y_true.shape[0], dtype='int8').dimshuffle(0, 'x', 'x')
    y_i = T.arange(y_true.shape[1], dtype='int8').dimshuffle('x', 0, 'x')
    z_i = y_predicted.argsort().astype('int8')
    reverse = [slice(None), slice(None), slice(None, None, -1)]
    y_true = y_true[x_i, y_i, z_i][reverse]
    y_predicted = y_predicted[x_i, y_i, z_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1], dtype='int8')
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _tensor4_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 4
    a_i = T.arange(y_true.shape[0], dtype='int8').dimshuffle(0, 'x', 'x', 'x')
    b_i = T.arange(y_true.shape[1], dtype='int8').dimshuffle('x', 0, 'x', 'x')
    c_i = T.arange(y_true.shape[2], dtype='int8').dimshuffle('x', 'x', 0, 'x')
    d_i = y_predicted.argsort().astype('int8')

    reverse = [slice(None), slice(None), slice(None), slice(None, None, -1)]
    y_true = y_true[a_i, b_i, c_i, d_i][reverse]
    y_predicted = y_predicted[a_i, b_i, c_i, d_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1], dtype='int8')
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _binary_clf_curves(y_true, y_predicted):
    """
    returns curves calculated axis[-1]-wise
    note - despite trying several approaches, could not seem to get a
    n-dimensional verision of clf_curve to work, so abandoning. 2,3,4 is fine.
    """
    print 'y_true', y_true
    print 'y_predicted', y_predicted
    print 'y_true.type', y_true.type
    print 'y_predicted.type', y_predicted.type
    print 'y_true.shape', y_true.shape
    print 'y_predicted.shape', y_predicted.shape
    print 'y_true.ndim', y_true.ndim
    print 'y_predicted.ndim', y_predicted.ndim
    if not (y_true.ndim == y_predicted.ndim):
        raise ValueError('Dimension mismatch, ({}, {})'.format(y_true.ndim, y_predicted.ndim))
    if not isinstance(y_true, T.TensorVariable) or not isinstance(y_predicted, T.TensorVariable):
        raise TypeError('This only works for symbolic variables.')

    if y_true.ndim == 1:
        clf_curve_fn = _vector_clf_curve
    elif y_true.ndim == 2:
        clf_curve_fn = _matrix_clf_curve
    elif y_true.ndim == 3: 
        clf_curve_fn = _tensor3_clf_curve
    elif y_true.ndim == 4:
        clf_curve_fn = _tensor4_clf_curve
    else:
        raise NotImplementedError('Not implemented for ndim {}'.format(y_true.ndim))

    fps, tps, thresholds = clf_curve_fn(y_true, y_predicted)
    return fps, tps, thresholds

def _last_col_idx(ndim):
    last_col = [slice(None) for x in xrange(ndim)]
    last_col[-1] = -1
    return last_col

def _reverse_idx(ndim):
    reverse = [slice(None) for _ in range(ndim-1)]
    reverse.append(slice(None, None, -1))
    return reverse

def roc_curves(y_true, y_predicted):
    "returns roc curves calculated axis -1-wise"
    fps, tps, thresholds = _binary_clf_curves(y_true, y_predicted)
    last_col = _last_col_idx(y_true.ndim)
    fpr = fps.astype('float32') / T.shape_padright(fps[last_col], 1)
    tpr = tps.astype('float32') / T.shape_padright(tps[last_col], 1)
    return fpr, tpr, thresholds

def roc_auc_scores(y_true, y_predicted):
    "roc auc scores calculated axis -1-wise"
    fpr, tpr, thresholds = roc_curves(y_true, y_predicted)
    return auc(fpr, tpr)

def roc_auc_loss(y_predicted, y_true):
    return 1-roc_auc_scores(y_true, y_predicted[:,2])

def roc_auc_loss(layers,
              loss_function,
              target,
              aggregate=None,
              **kwargs):
    output_layer = layers[-1]
    network_output = get_output(output_layer, **kwargs)
    print aggregate
    return 1-roc_auc_scores(target, network_output)
    # losses = loss_function(network_output, target)
    # return aggregate(losses)

def precision_recall_curves(y_true, y_predicted):
    "precision recall curves calculated axis -1-wise"
    fps, tps, thresholds = _binary_clf_curves(y_true, y_predicted)
    last_col = _last_col_idx(y_true.ndim)
    last_col[-1] = np.asarray([-1], dtype='int8')
    precision = tps.astype('float32') / (tps + fps)
    if y_true.ndim == 1:
        recall = tps.astype('float32') / tps[-1]
    else:
        recall = tps.astype('float32') / tps[last_col]
    reverse = _reverse_idx(fps.ndim)
    precision = precision[reverse]
    recall = recall[reverse]
    thresholds = thresholds[reverse]
    if y_true.ndim == 1:
        ones, zeros = np.asarray([1], dtype='float32'), np.asarray([0], dtype='float32')
    else:
        ones = T.ones_like(precision)[last_col]
        zeros = T.zeros_like(recall)[last_col]
    precision = T.concatenate([precision, ones], axis=-1) 
    recall = T.concatenate([recall, zeros], axis=-1)
    return precision, recall, thresholds

def average_precision_scores(y_true, y_predicted):
    precision, recall, _ = precision_recall_curves(y_true, y_predicted)
    return auc(recall, precision)

def precision_recall_loss(y_true, y_predicted):
    "convenience function to minimize for"
    return 1-average_precision_scores(y_true, y_predicted)

def last_axis_precision_recall_curve(y_true, y_predicted):
    fps, tps, thresholds = _last_axis_binary_clf_curve(y_true, y_predicted)
    i = [slice(None)] * fps.ndim
    i[-1] = [-1]
    precision = tps.astype('float32') / (tps + fps)
    recall = tps.astype('float32') / tps[i]
    i[-1] = slice(None, None, -1)
    precision = precision[i]
    recall = recall[i]
    thresholds = thresholds[i]
    i[-1] = [-1]
    precision = np.concatenate([precision, np.ones(precision.shape)[i]], axis=-1)
    recall = np.concatenate([recall, np.zeros(recall.shape)[i]], axis=-1)
    return precision, recall, thresholds



#aliases
roc_curve = roc_curves
roc_auc_score = roc_auc_scores
precision_recall_curve = precision_recall_curves
average_precision_score = average_precision_scores
_binary_clf_curve = _binary_clf_curves


#================ second objective ==================
def binary_crossentropy_with_ranking(layers,
              loss_function,
              target,
              aggregate=None,
              **kwargs):
    output_layer = layers[-1]
    network_output = get_output(output_layer, **kwargs)
    return bc_with_ranking(network_output, target)

_EPSILON = 10e-4

def bc_with_ranking(y_prediction, y_true):
    """ Trying to combine ranking loss with numeric precision"""
    y_pred = T.clip(y_prediction[:,1], _EPSILON, 1.-_EPSILON)
    # first get the log loss like normal
    logloss = T.mean(T.nnet.binary_crossentropy(y_pred, y_true), axis=0)
    
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = T.clip(y_pred, _EPSILON, 1.-_EPSILON)
    print y_pred_clipped

    # translate into the raw scores before the logit
    y_pred_score = T.log(y_pred_clipped / (1. - y_pred_clipped))
    print y_pred_score

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = T.max(y_pred_score * (y_true <1.))
    print y_pred_score_zerooutcome_max

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max
    print rankloss

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = T.sqr(T.clip(rankloss, -100., 0.))

    # average the loss for just the positive outcomes
    rankloss = T.sum(rankloss, axis=0) / (T.sum(y_true > 0.) + 1.)

    # return (rankloss + 1) * logloss - an alternative to try
    #return rankloss + logloss
    return logloss




#================ Turd objective ==================

MAX_INT = np.iinfo(np.int32).max


class JonasAUCobjective():

    #TODO: remove bias from 2 original values at infinity

    def __init__(self):
        """
        These must be kept sorted
        """
        self.T = theano.shared(value=np.array([-np.inf, np.inf], dtype='float32'), name='T')
        self.labels = theano.shared(value=np.array([0, 1], dtype='float32'), name='labels')
        self.TPR = theano.shared(value=np.array([1, 1], dtype='float32'), name='TPR')
        self.FPR = theano.shared(value=np.array([1, 0], dtype='float32'), name='FPR')
        self.time_added = np.array([MAX_INT, MAX_INT], dtype='int32')
        """
        These contain some global information on the previous lists
        """
        self.AUC0 = theano.shared(value=np.float32(1.), name='AUC0')
        self.N_P = theano.shared(value=np.float32(1.), name='N_P')
        self.N_F = theano.shared(value=np.float32(1.), name='N_F')
        """
        These contain some information for maintaining the lists
        """
        self.N_added = 2


    def auc_error(self, T_prediction, true_label):
        # get TPR1, TPR2, label_1, FPR1, T1, T2 from the estimated distribution
        idx = searchsorted(self.T, T_prediction, side='left')
        TPR1 = self.TPR[idx-1]
        TPR2 = self.TPR[idx]
        label_1 = self.labels[idx-1]
        label_2 = self.labels[idx]
        FPR1 = self.FPR[idx-1]
        T1 = self.T[idx-1]
        T2 = self.T[idx]

        f1 = 1 - label_1
        f2 = 1 - label_2
        l = true_label
        f = 1-true_label
        AUC1 = self.AUC0 + f1*l + f*TPR1 + l*(self.N_F-FPR1)
        dAUC = (TPR2 - TPR1)*f + f2*l

        # deal with new parameters at edge of list of parameters
        # note: they never have a gradient?
        coef = T.switch(T.isinf(T1), T.switch(T.isinf(T2), 0.5, 1.0),
                                     T.switch(T.isinf(T2), 0.0, (T_prediction-T1)/(T2-T1)))
        coef = (T_prediction-T1)/(T2-T1)
        AUCt = AUC1 + coef*dAUC

        return AUCt / ((self.N_P+l) * (self.N_F+f))


    def add_points(self, predicted, label, never_remove=False):
        idx = np.searchsorted(self.T.get_value(), predicted)
        self.T.set_value(np.insert(self.T.get_value(), idx, predicted,))
        labels = np.insert(self.labels.get_value(), idx, label,)
        self.labels.set_value(labels)
        self.TPR.set_value(np.cumsum(labels[::-1])[::-1])
        self.FPR.set_value(np.arange(1,len(labels)+1, dtype='float32')[::-1] - np.cumsum(labels[::-1])[::-1])
        if never_remove:
            self.time_added = np.insert(self.time_added, idx, MAX_INT)
        else:
            self.time_added = np.insert(self.time_added, idx, self.N_added)

        self.AUC0.set_value(np.sum((1-labels) * np.cumsum(labels[::-1])[::-1]))
        self.N_P.set_value(np.sum(labels))
        self.N_F.set_value(np.sum(1-labels))

        if isinstance(predicted, list):
            self.N_added += len(self.predicted)
        else:
            self.N_added += 1


    @property
    def current_auc(self):
        return self.AUC0.get_value() / (self.N_P.get_value() * self.N_F.get_value())


    def print_status(self):
        print "T:  \t", self.T.get_value()
        print "lbl:\t", self.labels.get_value()
        print "TPR:\t", self.TPR.get_value()
        print "FPR:\t", self.FPR.get_value()
        print "AUC0:\t", self.AUC0.get_value()
        print "N_P:\t", self.N_P.get_value()
        print "N_F:\t", self.N_F.get_value()
        print "AUC:\t", self.current_auc


    def __call__(self, *args, **kwargs):
        return self.auc_error(*args, **kwargs)

if __name__=="__main__":
    aucd = JonasAUCobjective()

    aucd.add_points(-1,1)
    aucd.add_points(3,1)
    aucd.print_status()


    w, x, y = T.scalar('w'), T.scalar('x'), T.scalar('y')
    z1 = aucd(w*x, y)
    f = theano.function([w, x, y], [z1], on_unused_input='ignore')
    print f(1,2,0)
    print f(1,2,1)

    gr1 = T.grad(z1, x)
    g = theano.function([w, x, y], [gr1], on_unused_input='ignore')
    print g(1,2,0)
    print g(1,2,1)




