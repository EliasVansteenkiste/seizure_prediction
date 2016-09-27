import numpy as np

class LearningRateScheduler(object):
    def __init__(self, schedule):
        self.schedule = schedule


    def __call__(self, nn, trainHistory):
        currentEpoch = trainHistory[-1]['epoch']
        nextEpoch = currentEpoch + 1
        if nextEpoch in self.schedule:
            nn.more_params['update_learning_rate'] = self.schedule[nextEpoch]

        