import numpy as np

class Model(BufferMixin):
    """
    Via beamformer, whole pipeline for estimation of masks is understood.
    Acoustic model is a generative model that filters the speech so it sounds
    more naturally.
    """
    def initialize(self):
        pass

    def process(self, sample):
        mask = self.beamformer(sample)
        pattern = self.acoustic_model(sample)
        return mask * sample + (1 - mask) * pattern * exp(1j * np.angle(sample))

