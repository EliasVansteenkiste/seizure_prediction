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
from lasagne.objectives import aggregate 
from theano.ifelse import ifelse


#================ first objective ==================
def binary_crossentropy_with_se_ranking(layers,
              loss_function,
              target,
              aggregate=None,
              **kwargs):
    output_layer = layers[-1]
    network_output = get_output(output_layer, **kwargs)
    return bc_with_ranking(network_output, target)

_EPSILON = 1e-6

def bc_with_se_ranking(y_prediction, y_true):
    """ Trying to combine ranking loss with numeric precision"""
    
    # first get the log loss like normal
    #logloss = aggregate(T.nnet.binary_crossentropy(y_pred, y_true))
    logloss = aggregate(T.nnet.categorical_crossentropy(y_prediction, y_true))
    
    y_pred = y_prediction[:,1]
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = T.clip(y_pred, _EPSILON, 1.-_EPSILON)

    # translate into the raw scores before the logit
    y_pred_score = T.log(y_pred_clipped / (1. - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = T.max(y_pred_score * (y_true <1.))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = T.sqr(T.clip(rankloss, -100., 0.))

    # average the loss for just the positive outcomes
    rankloss = T.sum(rankloss) / (T.sum(y_true > 0.) + 1.)

    # return (rankloss + 1) * logloss - an alternative to try
    #return rankloss + logloss
    return rankloss




#================ second objective ==================
def binary_crossentropy_with_ranking(layers,
              loss_function,
              target,
              aggregate=None,
              **kwargs):
    output_layer = layers[-1]
    network_output = get_output(output_layer, **kwargs)
    return bc_with_ranking(network_output, target)

_EPSILON = 1e-6

def bc_with_ranking(y_prediction, y_true):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    #logloss = aggregate(T.nnet.binary_crossentropy(y_pred, y_true))
    
    # clip the probabilities to keep stability
    y_pred_clipped = T.clip(y_prediction, _EPSILON, 1.-_EPSILON)

    logloss = aggregate(T.nnet.categorical_crossentropy(y_prediction, y_true))
    
    y_pred = y_pred_clipped[:,1]
    # next, build a rank loss

    # translate into the raw scores before the logit
    y_pred_score = T.log(y_pred / (1. - y_pred))


    # determine what the maximum score for a zero outcome is
    max_zerooutcome = T.max(y_pred_score * (y_true <1.))

    mean_oneoutcome = T.mean(y_pred_score * (y_true > 0.1))

    border = ifelse(T.gt(max_zerooutcome, mean_oneoutcome), mean_oneoutcome, max_zerooutcome)

    # determine how much each score is above or below it
    rankloss = y_pred_score - border

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = T.sqr(T.clip(rankloss, -100., 0.))

    # average the loss for just the positive outcomes
    rankloss = T.sum(rankloss) / (T.sum(y_true > 0.1) + 1.)



    # determine what the maximum score for a zero outcome is
    min_oneoutcome = T.min(y_pred_score * (y_true > 0.1))

    mean_zerooutcome = T.mean(y_pred_score * (y_true < 1.))

    border = ifelse(T.lt(min_oneoutcome, mean_zerooutcome), mean_zerooutcome, min_oneoutcome)

    # determine how much each score is above or below it
    rankloss_ = y_pred_score - border

    # only keep losses for positive outcomes
    rankloss_ = rankloss_ * (1. - y_true)

    # only keep losses where the score is below the max
    rankloss_ = T.sqr(T.clip(rankloss_, 0., 100.))

    # average the loss for just the positive outcomes
    rankloss_ = T.sum(rankloss_, axis=0) / (T.sum(y_true < 1.) + 1.)

    # return (rankloss + 1) * logloss - an alternative to try
    #return rankloss + logloss
    return logloss #0.01*rankloss_ #+ 0.01*rankloss_ #+ logloss


#================ Turd objective ==================

MAX_INT = np.iinfo(np.int32).max


class jonas_auc_objective():

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

    def add_batch():
        output_layer = layers[-1]
        network_output = get_output(output_layer, **kwargs)
        for output, label in zip(network_output,target):
            self.add_points(output,label)

    def __call__(self,layers,
              loss_function,
              target,
              aggregate=None,
              **kwargs):
        output_layer = layers[-1]
        network_output = get_output(output_layer, **kwargs)
        return T.mean(self.auc_error(network_output, target))

if __name__=="__main__":
    aucd = jonas_auc_objective()

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


