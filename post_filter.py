import numpy as np

from utils import BufferMixin

class PostFilter(BufferMixin([1, 257], np.complex64)):
    def initialize(self):
        pass

    def process(self, sample):
        pass
