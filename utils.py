import numpy as np

def BufferMixin(buffer_size=[1, 257], dtype=np.float32):


    class BufferMixinClass:
        """
        Adds a buffer that can be used to hold windows of time
        (useful e.g. for convolution in time)
        """
    
        class Buffer(np.ndarray):
            def __init__(self, *args, **kwargs):
                assert len(args[0]) > 1
                self[:] = 0
    
            def push(self, sample):
                self[:-1] = self[1:]
                self[-1] = sample
    
        def __init__(self, buffer_size=buffer_size, dtype=dtype):
            self.buffer = Buffer(buffer_size, dtype)

    return BufferMixinClass


class Remix:
    def __init__(self):
        self.buffer = np.zeros(512, np.float32)

    def process(self, sample):
        self.buffer[:(512-128)] = self.buffer[128:]
        self.buffer[(512-128):] = 0
        self.buffer += ifft(sample)
        return self.buffer[:128].copy()

def fft(x):
    return np.hamming(512)**0.5 * np.fft.rfft(x)

def ifft(x):
    return np.hamming(512)**0.5 * np.fft.irfft(x)
