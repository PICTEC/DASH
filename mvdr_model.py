import itertools
import keras
import keras.backend as K
import math
import numpy as np
import datetime


class Model:
    def __init__(self, n, frame_len, delay_and_sum, use_channels, model_name):
        self.model_path = model_name
        self.mask_thresh_speech = 0.7
        self.mask_thresh_noise = 0.3
        self.num_of_mics = n
        self.delay_and_sum = delay_and_sum # TO DO
        self.use_channels = use_channels  # TO DO
        self.frame_len = frame_len
        self.psd_tracking_constant_speech = 0.5 + 0j
        self.psd_tracking_constant_noise = 0.99 + 0j
        self.frame = 0
        self.fft_len = int(self.frame_len / 2 + 1)
        self.eigenvector = np.ones((self.fft_len, self.num_of_mics), dtype=np.complex64) +\
                           np.zeros((self.fft_len, self.num_of_mics), dtype=np.complex64) * 1j
        self.psd_speech = np.tile(np.diag(np.ones(self.num_of_mics)), (self.fft_len, 1)).reshape(-1, self.num_of_mics, self.num_of_mics).astype(np.complex64)
        self.psd_noise = np.tile(np.diag(np.ones(self.num_of_mics)), (self.fft_len, 1)).reshape(-1, self.num_of_mics, self.num_of_mics).astype(np.complex64)

    def fast_mvdr(self, sound, steervect):
        cminv = np.linalg.inv(self.psd_noise)
        conj = np.conj(steervect).reshape(self.fft_len, 1, -1)
        return (conj @ cminv @ sound.reshape(self.fft_len, -1, 1)) / (
                conj @ cminv @ steervect.reshape(self.fft_len, -1, 1))

    def update_psds(self, fft_vector, speech_mask, noise_mask):
        toUpd = speech_mask
        self.psd_speech[toUpd] = self.psd_tracking_constant_speech * self.psd_speech[toUpd] + \
                                 (1 - self.psd_tracking_constant_speech) * \
                                 np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]
        toUpd = noise_mask
        self.psd_noise[toUpd] = self.psd_tracking_constant_noise * self.psd_noise[toUpd] + \
                                (1 - self.psd_tracking_constant_noise) * \
                                np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]

    def update_ev_by_power_iteration(self):
        # Uncomment to use non-working method of estimation without decomposition :P
        # unnormalized_eigenvector = np.einsum('...ij,...j->...i', self.psd_speech, self.eigenvector, dtype=np.complex128)
        # self.eigenvector = unnormalized_eigenvector / np.linalg.norm(unnormalized_eigenvector)
        self.eigenvector = np.linalg.eig(self.psd_speech)[0]

    def initialize(self):
        self.model = keras.models.load_model(self.model_path)
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()
        # Three dry run to compile this magical device
        for i in range(3):
            prep = np.random.random([8, 1, 257]).astype(np.float32)
            self.session.run(self.output,
                feed_dict={self.input: prep})


    def process(self, ffts):
        prep = ffts.T.reshape(self.num_of_mics, 1, -1)
        prep = np.abs(prep)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        vad_mask = np.transpose(np.clip(response, 0, None) ** 3, [2, 0, 1])
        speech_update = (vad_mask > self.mask_thresh_speech) * (vad_mask > self.mask_thresh_speech).transpose([0, 2, 1])
        speech_update = speech_update.sum((1,2))
        noise_update = (vad_mask < self.mask_thresh_noise) * (vad_mask < self.mask_thresh_noise).transpose([0, 2, 1])
        noise_update = noise_update.sum((1,2))
        self.update_psds(ffts, speech_update, noise_update)
        self.update_ev_by_power_iteration()
        result_fftd = self.fast_mvdr(ffts, self.eigenvector)
        return result_fftd.reshape(-1, 1)
