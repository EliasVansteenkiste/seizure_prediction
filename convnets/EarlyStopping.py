import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.bestValid = np.inf
        self.bestValidEpoch = 0
        self.bestWeights = None

    def __call__(self, nn, trainHistory):
        currentValid = trainHistory[-1]['valid_loss']
        currentEpoch = trainHistory[-1]['epoch']
        if currentValid < self.bestValid:
            self.bestValid = currentValid
            self.bestValidEpoch = currentEpoch
            self.bestWeights = nn.get_all_params_values()
        elif self.bestValidEpoch + self.patience < currentEpoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.bestValid, self.bestValidEpoch))
            nn.load_params_from(self.bestWeights)
            raise StopIteration()