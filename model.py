from keras.models import load_model
import numpy as np
import tensorflow as tf

from utils import BufferMixin


class ReferenceModelInterface:
    def __init__(self):
        raise NotImplementedError

    def initialize(self, **kwargs):
        raise NotImplementedError

    def process(self, sample):
        raise NotImplementedError


class DAEAcousticModel:
    def __init__(self, model_path, ratio=1.0):
        self.model = load_model(model_path, {"tf": tf})

    def __call__(self, sample):
        phaze = np.angle(sample)
        sample = np.log(np.abs(sample) ** 2 + 2e-12)
        reconstruction = self.model(sample)
        a = (sample[:, :, 1:] / reconstruction[:, :, 1:]).mean()
        a *= self.ratio
        return np.sqrt(np.exp(a * reconstruction)) * np.exp(1j * phaze)


class DolphinModel(BufferMixin([17, 257], np.complex64)):
    """
    Via beamformer, whole pipeline for estimation of masks is understood.
    Acoustic model is a generative model that filters the speech so it sounds
    more naturally.
    """
    def initialize(self, am_mode='dae', am_path='bin/am-dae.h5'):
        if am_mode == 'dae':
            self.acoustic_model = DAEAcousticModel(am_path)

    def process(self, sample):
        self.buffer.push(sample)
        mask = self.beamformer(self.buffer)
        pattern = self.acoustic_model(self.buffer)
        return mask * sample + (1 - mask) * pattern * np.exp(1j * np.angle(sample))
