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
        self.ratio = ratio

    def __call__(self, sample):
        phaze = np.angle(sample)
        sample = np.log(np.abs(sample) ** 2 + 2e-12)
        sample = np.stack([sample[:, :, i] for i in range(sample.shape[2])])
        reconstruction = self.model.predict(sample)
        a = (sample[:, 16, :] / reconstruction[:, 0, :]).mean()
        a *= self.ratio
        toplay = np.sqrt(np.exp(a * reconstruction[:, 0, :].mean(0))) # * np.exp(1j * phaze[16, :, 0]) # should phase be here?
        return toplay


class DolphinModel(BufferMixin([17, 257, 6], np.complex64)):
    """
    Beamformer is a whole estimation of masks pipeline.
    Acoustic model is a generative model that filters the speech so it sounds
    more naturally.
    """
    def initialize(self, am_mode='dae', am_path='bin/am-dae.h5'):
        if am_mode == 'dae':
            self.acoustic_model = DAEAcousticModel(am_path)
        self.beamformer = lambda x: np.ones([257])

    def process(self, sample):
        self.buffer.push(sample)
        mask = self.beamformer(self.buffer)
        pattern = self.acoustic_model(self.buffer)
        ret = (mask * sample[:, 0] + (1 - mask) * pattern * np.exp(1j * np.angle(sample[:, 0]))).reshape(-1, 1)
        return ret
