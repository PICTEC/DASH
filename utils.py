import numpy as np
import os
import scipy.io.wavfile as sio
import subprocess
import tempfile


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


def open_sound(fname):
    """
    Opens flac and wav files
    """
    if fname.endswith(".flac"):
        file = tempfile.mktemp() + ".wav"
        subprocess.Popen(["sox", fname, file]).communicate()
        sr, data = sio.read(file)
        os.remove(file)
        data = data.astype(np.float32)
        if np.any(data > 1):
            data /= 2**15
        return sr, data
    sr, data = sio.read(fname)
    data = data.astype(np.float32)
    if np.any(data > 1):
        data /= 2**15
    return sr, data


def list_sounds(src):
    sources = []
    for path, dirs, fnames in os.walk(src):
        for fname in fnames:
            if fname.endswith(".wav") or fname.endswith(".flac"):
                sources.append(os.path.join(path, fname))
    return sources
