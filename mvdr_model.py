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
        self.mask_thresh = 0.5
        self.num_of_mics = n
        self.speed_of_sound = speed_of_sound
        self.frame_hop = frame_hop
        self.frame_len = frame_len
        self.frequency = f
        self.mu_cov = float(mu_cov)
        self.psd_tracking_constant_speech = 0.975 + 0j
        self.psd_tracking_constant_noise = 0.975 + 0j
        self.frame = 0
        self.fft_len = int(self.frame_len / 2 + 1)
        self.eigenvector = np.random.rand(self.fft_len, self.num_of_mics)/self.num_of_mics +\
                               np.random.rand(self.fft_len, self.num_of_mics)/self.num_of_mics*1j
        self.psd_speech = np.random.random((self.fft_len, self.num_of_mics, self.num_of_mics)).astype(np.float32)
        self.psd_noise = np.random.random((self.fft_len, self.num_of_mics, self.num_of_mics)).astype(np.float32)
        self.spat_cov_mat = np.zeros((self.fft_len, self.num_of_mics, self.num_of_mics), dtype=np.complex64)

    def estimate_covariance_mat(self, mask, signal):
        sig = signal.reshape(self.fft_len, -1, 1) @ np.conj(signal).reshape(self.fft_len, 1, -1)
        update = self.mu_cov * self.spat_cov_mat + (1 - self.mu_cov) * sig
        return (1 - mask) * self.spat_cov_mat + mask * update

    def fast_mvdr(self, sound, steervect):
        cminv = np.linalg.inv(self.spat_cov_mat)
        conj = np.conj(steervect).reshape(self.fft_len, 1, -1)
        return (conj @ cminv @ sound.reshape(self.fft_len, -1, 1)) / (
                conj @ cminv @ steervect.reshape(self.fft_len, -1, 1))

    def update_psds(self, fft_vector, speech_mask):
        # which PSDs will be updated
        toUpd = speech_mask > self.mask_thresh
        self.psd_speech[toUpd] = self.psd_tracking_constant_speech * self.psd_speech[toUpd] + \
                                 (1 - self.psd_tracking_constant_speech) * \
                                 np.einsum('ij,ik->ijk', fft_vector, fft_vector.conj())[toUpd]

        toUpd = np.invert(toUpd)
        self.psd_noise[toUpd] = self.psd_tracking_constant_noise * self.psd_noise[toUpd] + \
                                (1 - self.psd_tracking_constant_noise) * \
                                np.einsum('ij,ik->ijk', fft_vector, fft_vector.conj())[toUpd]

    def update_ev_by_power_iteration(self):
        unnormalized_eigenvector = np.einsum('...ij,...j->...i', self.psd_speech, self.eigenvector, dtype=np.complex128)
        print(self.psd_speech.dtype)
        print(self.eigenvector.dtype)
        print(unnormalized_eigenvector.dtype)
        self.eigenvector = unnormalized_eigenvector / \
                      np.sqrt(np.sum(np.abs(unnormalized_eigenvector)) ** 2 / self.num_of_mics)

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
        vad_mask = np.transpose(np.clip(response, 0, None) ** 1.5, [2, 0, 1])
        vad_mask = vad_mask * vad_mask.transpose([0, 2, 1])
        self.spat_cov_mat = self.estimate_covariance_mat(vad_mask, ffts)
        self.update_psds(ffts, vad_mask)
        self.update_ev_by_power_iteration()
        # print(self.eigenvector[:,0])
        result_fftd = self.fast_mvdr(ffts, self.eigenvector)
        print(result_fftd)
        return result_fftd.reshape(-1, 1)
