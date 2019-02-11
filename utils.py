from keras.callbacks import Callback
import numpy as np
import os
import scipy.io.wavfile as sio
import subprocess
import tempfile
import time

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
            self.buffer = BufferMixinClass.Buffer(buffer_size, dtype)

    return BufferMixinClass


class Remix:
    def __init__(self, buffer_size, buffer_hop, channels):
        self.buffer_size = buffer_size
        self.buffer_hop = buffer_hop
        self.channels = channels
        self.buffer = np.zeros((buffer_size, channels), np.float32)

        self.overlaps = self.buffer_size / self.buffer_hop

    def process(self, sample):
        self.buffer = np.roll(self.buffer, -self.buffer_hop, axis=0)
        self.buffer[-self.buffer_hop:,:] = 0
        r = ifft(sample, self.buffer_size, self.channels)
        if self.channels == 1:
            self.buffer[:,0] += r
        else:
            self.buffer += r
        return self.buffer[:self.buffer_hop].copy() / self.overlaps


def stft(y, n_fft=512, hop_length=128, window=np.hamming):
    if window is not None:
        win_val = window(n_fft)
    else:
        win_val = np.ones(n_fft, np.float32)
    how_many = (len(y) - n_fft) // hop_length
    fft_size = n_fft // 2 + 1
    stft_v = np.zeros([fft_size, how_many], np.complex64)
    for i in range(how_many):
        win = y[hop_length * i : hop_length * i + n_fft]
        win = win * win_val
        win = np.fft.rfft(win)
        stft_v[:, i] = win
    return stft_v

WINDOW = None
FRAMEW = None
CHANNELS = None

# TODO: this should be object now...
def fft(x, frame_width, channels):
    global FRAMEW, WINDOW, CHANNELS
    if FRAMEW != frame_width:
        FRAMEW = frame_width
        CHANNELS = channels
        WINDOW = np.stack([np.hamming(frame_width) ** 0.5] * channels).T
    return np.fft.rfft(WINDOW * x, axis=0)


def ifft(x, frame_width, channels):
    out = np.zeros((frame_width, channels), dtype=np.float32)
    if channels == 1:
        out =  np.hamming(frame_width)**0.5 *np.fft.irfft(x)
    else:
        for ch in range(channels):
            out[:, ch] = np.fft.irfft(x[:, ch]) * np.hamming(frame_width) ** 0.5
    return out


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


class StopOnConvergence(Callback):
    def __init__(self, max_repetitions=10):
        super().__init__()
        self.max_repetitions = max_repetitions

    def on_train_begin(self, logs=None):
        self.repetitions = 0
        self.last_loss = np.inf

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if loss > self.last_loss:
                self.repetitions += 1
            else:
                self.last_loss = loss
                self.repetitions = 0
            if self.repetitions > self.max_repetitions:
                self.model.stop_training = True

def save_model(model, path):
    """
    Model should be stripped of all callbacks and needless objects...
    """
    model.optimizer = None
    model.built = False
    model.loss = None
    model.save(path)
