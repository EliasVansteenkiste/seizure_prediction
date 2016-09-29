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
from LearningRateScheduler import LearningRateScheduler
# Add the following line to the nolearn parameter set
# on_epoch_finished=[EarlyStopping(patience=4),]


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


def net0(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net


def net1(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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
        update_learning_rate=0.0005,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net


def net2(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
    layer = InputLayer(shape=(None, n_channels, width, height))
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(33,3), pad=(16,1), num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(33,3), pad=(16,1), num_filters=64)
    layer = MaxPool2DLayer(layer, pool_size=(2,2))
    layer = dropout(layer,p=0.5)
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
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

def net3(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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
        update_learning_rate=0.0001,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net


def net4(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

def net5(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
    layer = InputLayer(shape=(None, n_channels, width, height))
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, stride=(2,2), filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, stride=(2,2), filter_size=(3,3), pad=1, num_filters=64)
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
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

def net6(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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

    num_epochs_train = 60
    base_lr = 0.0001 
    learning_rate_schedule = { 0: base_lr, 7*num_epochs_train/3: base_lr/10, 9*num_epochs_train/10: base_lr/9, }
    

    net = NeuralNet(
        layer,
        update=adam,
        update_learning_rate=base_lr,
        on_epoch_finished=[LearningRateScheduler(learning_rate_schedule),],
        regression=False,
        max_epochs=num_epochs_train,
        verbose=1,
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

def net7(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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
        update_learning_rate=0.0001,
        #update_momentum=0.9,
        regression=False,
        max_epochs=100,
        verbose=1,
        on_epoch_finished=[EarlyStopping(patience=5),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

def net8(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
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

    num_epochs_train = 60
    base_lr = 0.0001 
    learning_rate_schedule = { 0: base_lr, 6: base_lr/10, }
    

    net = NeuralNet(
        layer,
        update=adam,
        update_learning_rate=base_lr,
        on_epoch_finished=[LearningRateScheduler(learning_rate_schedule),],
        regression=False,
        max_epochs=num_epochs_train,
        verbose=1,
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net


def net9(n_channels,width,height,n_output=2,nonlinearity=nonlinearities.very_leaky_rectify,batch_iterator_train=BatchIterator(batch_size=256),batch_iterator_test=BatchIterator(batch_size=256)):   
    layer = InputLayer(shape=(None, n_channels, width, height))
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, stride=(2,2), filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, stride=(2,2), filter_size=(3,3), pad=1, num_filters=64)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, filter_size=(3,3), pad=1, num_filters=64)
    layer = dropout(layer,p=0.5)
    layer = Conv2DLayer(layer, nonlinearity=nonlinearity, stride=(2,2), filter_size=(3,3), pad=1, num_filters=64)
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
        on_epoch_finished=[EarlyStopping(patience=4),],
        batch_iterator_train = batch_iterator_train,
        batch_iterator_test = batch_iterator_test,
    )
    return net

