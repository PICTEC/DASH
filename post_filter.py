from keras.models import load_model
import numpy as np

from utils import BufferMixin, StopOnConvergence
import dae


class DAEPostFilter(BufferMixin([17, 257], np.complex64)):
    """
    It is assumed that PostFilter preserves phase of original signal.
    """

    _all_imports = {}
    _all_imports.update(dae.imports)

    def __init__(self, fname=None):
        self.model = load_model(fname, self._all_imports)
        # TODO: perform checks whether fft_bin_size is proper

    def initialize(self):
        pass

    # TODO: this given an ENORMOUS shift in buffers - to reimplement
    def process(self, sample):
        self.buffer.push(sample)
        predictive = np.log(np.abs(self.buffer.reshape([1, 17, 257])) ** 2)
        result = self.model.predict(predictive)
        result = result[0, 8, :]  # extract channel of interest
        result = result * np.exp(1j * np.angle(sample))  # restore phase information
        return result

    @staticmethod
    def train(model_config, trainX, trainY, valid_ratio=0.1, ):
        """
        This should create a model from some training script...
        """
        spec = ... # get model spec somehow
        model = spec()
        # prepare validation and training data
        model.train()
        


class NullPostFilter:
    def initialize(self):
        pass

    def process(self, sample):
        return sample
