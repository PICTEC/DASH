from keras.models import load_model
import numpy as np

from utils import BufferMixin


class DAEAcousticModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def __call__(self, sample):
        reconstruction = self.model(sample)
        a = (sample[:, :, 1:] / reconstruction[:, :, 1:]).mean()
        return a * reconstruction


class Model(BufferMixin([17, 257], np.complex64)):
    """
    Via beamformer, whole pipeline for estimation of masks is understood.
    Acoustic model is a generative model that filters the speech so it sounds
    more naturally.
    """
    def initialize(self, am_mode='dae', am_path='storage/model-dae.h5'):
        if am_mode == 'dae':
            self.acoustic_model = DAEAcousticModel(am_path)

    def process(self, sample):
        self.buffer.push(sample)
        mask = self.beamformer(self.buffer)
        pattern = self.acoustic_model(self.buffer)
        return mask * sample + (1 - mask) * pattern * np.exp(1j * np.angle(sample))
