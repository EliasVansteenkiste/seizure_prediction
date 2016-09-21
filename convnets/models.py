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

def net1(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
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
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,1))
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

def net2(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
    layer = InputLayer(shape=(None, n_channels, width, height), name='input')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(32,1), pad=1, num_filters=64, name='conv1')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv2')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv3')
    layer = MaxPool2DLayer(layer, pool_size=(2,2), name='maxpool1')
    layer = dropout(layer,p=0.5, name='dropout1')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv4')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv5')
    layer = MaxPool2DLayer(layer, pool_size=(2,2), name='maxpool1')
    layer = dropout(layer,p=0.5, name='dropout2')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv6')
    layer = Conv2DLayer(layer, nonlinearity=None, filter_size=(3,3), pad=1, num_filters=n_output, name='conv2output')
    layer = GlobalPoolLayer(layer, name='globalpool')
    layer = NonlinearityLayer(layer,nonlinearity=nonlinearities.softmax, name='softmax')

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

def net3(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
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
        update_learning_rate=0.01,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=5),],
    )
    return net

def net4(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
    layer = InputLayer(shape=(None, n_channels, width, height), name='input')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv1')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv2')
    layer = MaxPool2DLayer(layer, pool_size=(2,2), name='maxpool1')
    layer = dropout(layer,p=0.5, name='dropout1')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv3')
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64, name='conv4')
    layer = MaxPool2DLayer(layer, pool_size=(2,2), name='maxpool2')
    layer = dropout(layer,p=0.5, name='dropout1')
    layer = DenseLayer(layer, nonlinearity=nonlinearity, num_units=256)
    layer = batch_norm(layer)
    layer = dropout(layer,p=0.5)
    layer = DenseLayer(layer, nonlinearity=nonlinearity, num_units=256)
    layer = batch_norm(layer)
    layer = dropout(layer,p=0.5)
    layer = DenseLayer(layer, nonlinearity=nonlinearities.softmax, num_units=n_output)

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

def net5(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify):   
    layer = InputLayer(shape=(None, n_channels, width, height))
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,2))
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,2))
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), stride=(4,1), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), stride=(4,1), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=None, filter_size=(3,3), pad=1, num_filters=n_output)
    layer = GlobalPoolLayer(layer)
    layer = NonlinearityLayer(layer,nonlinearity=nonlinearities.softmax)

    net = NeuralNet(
        layer,
        update=adam,
        update_learning_rate=0.01,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=5),],
    )
    return net

