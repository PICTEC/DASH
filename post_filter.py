from keras.models import load_model
import numpy as np

from utils import BufferMixin
import dae


class PostFilter(BufferMixin([17, 257], np.complex64)):

    _all_imports = {}
    _all_imports.update(dae.imports)

    def initialize(self, mode='dae', fname=None):
        self.model = load_model(fname, self._all_imports)

    # TODO: this given an ENORMOUS shift in buffers - to reimplement
    def process(self, sample):
        self.buffer.push(sample)
        result = self.model.predict(self.buffer.reshape([1, 17, 257]))
        return result[0, 8]
