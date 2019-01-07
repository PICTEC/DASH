import numpy as np

class BufferMixin:
    def push(self, sample):
        ...

    def get(self, sample):
        ...

def fft(x):
    return np.hamming(512)**0.5 * np.fft.rfft(x)

def ifft(x):
    return np.hamming(512)**0.5 * np.fft.irfft(x)
