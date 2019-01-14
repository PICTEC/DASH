import keras
import numpy as np
import scipy.stats as sst


class DataExtractor:
    def transform(self):
        """
        Need a file list and specification
        """
        self.n_channels


class GMMModel(object):
    def __init__(self):
        pass

    def build(self):
        norm_act = lambda x: BatchNormalization()(LeakyReLU(0.01)(x))
        input = Input((None, 257, n_channel))
        chmix = norm_act(Conv2D(32, 1, activation='relu')(input))
        chmix = norm_act(Conv2D(12, 1, activation='relu')(chmix))
        chmix = norm_act(Conv2D(2, 1, activation='relu')(chmix))
        flatten = TimeDistributed(Flatten()(chmix))
        dense = norm_act(Dense(...)(flatten))
        dense = Dense(n_mixt * 3)(dense)
        output = Reshape((None, n_mixt, 3))(dense)
        self.model = Model(input, output)

    def train(self, X, Y):
        pass

    def integrate(self, observation):
        parameters = self.model.predict(observation)
        expects = lambda X: [np.sum(parameters[i, :, 0] * [
                sst.norm(loc=parameters[i, ix, 1], scale=parameters[i, ix, 2]).expect(lambda x:x, ub=x)
                for ix in range(parameters.shape[1])
                ]) for i, x in enumerate(X)]
        cdfs = lambda X: [np.sum(parameters[i, :, 0] * [
                sst.norm(loc=parameters[i, ix, 1], scale=parameters[i, ix, 2]).cdf(x)
                for ix in range(parameters.shape[1])
                ]) for i, x in enumerate(X)]
        return expects(observation) / cdfs(observation)


if __name__ == "__main__":
    pass
