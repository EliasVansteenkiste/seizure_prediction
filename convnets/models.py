import lasagne
from lasagne import layers
from lasagne.layers import *
from lasagne import nonlinearities
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import theano
import numpy as np
from lasagne.init import HeNormal
import sys

from EarlyStopping import EarlyStopping
# Add the following line to the nolearn parameter set
# on_epoch_finished=[EarlyStopping(patience=5),]


def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)


def net0(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
    layer = InputLayer(shape=(None, n_channels, width, height))
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,2))
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,2))
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=None, filter_size=(3,3), pad=1, num_filters=n_output)
    layer = GlobalPoolLayer(layer)
    layer = NonlinearityLayer(layer,nonlinearity=nonlinearities.softmax)

    net = NeuralNet(
        layer,
        update=adam,
        update_learning_rate=0.001,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=5),],
    )
    return net

