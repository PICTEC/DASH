import itertools
import keras
import keras.backend as K
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model:
    def __init__(self, n, f, speed_of_sound, mics_locs, frame_hop, frame_len, mu_cov):
        self.mask_thresh_speech = 0.7
        self.mask_thresh_noise = 0.3
        self.num_of_mics = n
        self.speed_of_sound = speed_of_sound
        self.frame_hop = frame_hop
        self.frame_len = frame_len
        self.frequency = f
        self.mu_cov = float(mu_cov)
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

    def update_psds(self, fft_vector, speech_mask):
        # which PSDs will be updated
        toUpd = speech_mask > self.mask_thresh_speech
        self.psd_speech[toUpd] = self.psd_tracking_constant_speech * self.psd_speech[toUpd] + \
                                 (1 - self.psd_tracking_constant_speech) * \
                                 np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]
        # print(self.psd_speech[2,:,:])
        toUpd = speech_mask < self.mask_thresh_speech
        self.psd_noise[toUpd] = self.psd_tracking_constant_noise * self.psd_noise[toUpd] + \
                                (1 - self.psd_tracking_constant_noise) * \
                                np.einsum('...i,...j->...ij', fft_vector, fft_vector.conj())[toUpd]
    def update_ev_by_power_iteration(self):
        # unnormalized_eigenvector = np.einsum('...ij,...j->...i', self.psd_speech, self.eigenvector, dtype=np.complex128)
        # self.eigenvector = unnormalized_eigenvector / np.linalg.norm(unnormalized_eigenvector)
        self.eigenvector = np.linalg.eig(self.psd_speech)[0]

    def initialize(self):
        self.model = keras.models.load_model("storage/8chmask.h5")
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()

    def process(self, ffts):
        prep = ffts.T.reshape(self.num_of_mics, 1, -1)
        prep = np.abs(prep)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        # vad_mask = np.transpose(np.clip(response, 0, None) ** 1.5, [2, 0, 1])
        # vad_mask = vad_mask * vad_mask.transpose([0, 2, 1])
        # self.update_psds(ffts, vad_mask)
        # self.update_ev_by_power_iteration()
        d_theta = np.ones((self.fft_len, self.num_of_mics), dtype=np.complex64)
        factor_1 = -1j * 2 * np.pi
        for k in range(1, self. fft_len):
            factor_2 = (k * self.frequency) / (self.frame_len / 2)
            d_theta[k, :] = [1,
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2),
                np.exp(factor_1 * 0 / factor_2)]
        # result_fftd = self.fast_mvdr(ffts, d_theta)
        # result_fftd = self.fast_mvdr(ffts, self.eigenvector)
        result_fftd = (ffts * d_theta).sum(1).reshape(-1,1)/self.num_of_mics
        return result_fftd.reshape(-1, 1)

    def delay_and_sum(self, ftts):
        steering_vector = ...
        return (steering_vecotr * ftts).sum(1).reshape(-1, 1)
