import scipy.io.wavfile as sio
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from joblib import Parallel, delayed
import multiprocessing


# kwargs = {"n": 6, "f": 16000, "speed_of_sound": 340, "frame_hop": 128, "frame_len": 512, "mu_cov": 0.95,
#           "mics_locs": [[0.00000001, 0.00000001, 0.00000001],
#                         [0.1, 0.00000001, 0.00000001],
#                         [0.2, 0.00000001, 0.00000001],
#                         [0.00000001, -0.19, 0.00000001],
#                         [0.1, -0.19, 0.00000001],
#                         [0.2, -0.19, 0.00000001]]}


class Model:
    class VAD:  # TODO this needs to be tested, especially if spec_avg works properly
        def __init__(self, mu_vad, init_est_vad, frame_len, vad_thresh):
            self.mu_vad = mu_vad
            self.init_est_vad = init_est_vad
            self.vad_thresh = vad_thresh
            self.spec_avg = np.ones(int(frame_len / 2 + 1))

        def vad(self, fftd):
            fftd += 0.00000001
            whitened_spec = (fftd / self.spec_avg) ** 2
            self.spec_avg = self.spec_avg * self.mu_vad + fftd * (1 - self.mu_vad)
            p = whitened_spec / np.sum(whitened_spec)
            return -np.sum(p * np.log(p))

    # TODO make DOA more classy
    class DOA:
        def __init__(self):
            self.azimuth = np.pi / 2
            self.elevation = np.pi / 2
            # pass

        def gcc_phat(self, sigl_fft, sigr_fft):
            sigr_fft_star = np.conj(sigr_fft)
            cc = sigl_fft * sigr_fft_star
            cc_phat = cc / abs(cc)
            r_phat = sfft.irfft(cc_phat)
            return r_phat

        def combine_gccs(self, mat, angles_list, results_array, combs_list, angle_matrix):
            x = np.linspace(0, 180, 180)
            alpha = np.zeros_like(x)
            beta = np.zeros_like(x)
            # for now this is very specific to current geometry
            # working rotation on xy axis
            for combo in combs_list:
                full_combo = mat.all_combs[combo]
                if np.round(mat.angle_matrix[full_combo[0], full_combo[1], 0] * 180 / np.pi) != 90:
                    # rotate results
                    rotated_angles = angles_list[combo] - angle_matrix[mat.all_combs[combo]][0]
                    rotated_results = np.append(results_array[combo][np.argwhere(rotated_angles < 0)],
                                                results_array[combo][np.argwhere(rotated_angles > 0)])
                    rotated_angles = np.append(rotated_angles[np.argwhere(rotated_angles < 0)] + np.pi,
                                               rotated_angles[np.argwhere(rotated_angles > 0)])
                    #
                    distances = ((np.asarray(np.append(np.pi, rotated_angles)) / np.pi * 180)[0:-1] -
                                 (np.asarray(np.append(np.pi, rotated_angles)) / np.pi * 180)[1:])[
                                0:int(np.ceil(len(rotated_angles) / 2))]
                    distances = np.append(distances, distances[0:-1][::-1])
                    single = np.zeros_like(x)
                    for peak in range(0, len(angles_list[combo])):
                        mu = rotated_angles[peak] / np.pi * 180
                        variance = 1.5 * distances[peak]
                        sigma = math.sqrt(variance)
                        single += mlab.normpdf(x, mu, sigma) * rotated_results[peak]
                    alpha += single
                    # plt.plot(x, single)
                if np.round(mat.angle_matrix[full_combo[0], full_combo[1], 0] * 180 / np.pi) != 0:
                    # rotate results
                    rotated_angles = angles_list[combo] - angle_matrix[mat.all_combs[combo]][0] - np.pi / 2
                    rotated_results = np.append(results_array[combo][np.argwhere(rotated_angles < 0)],
                                                results_array[combo][np.argwhere(rotated_angles > 0)])
                    rotated_angles = np.append(rotated_angles[np.argwhere(rotated_angles < 0)] + np.pi,
                                               rotated_angles[np.argwhere(rotated_angles > 0)])
                    #
                    distances = ((np.asarray(np.append(np.pi, rotated_angles)) / np.pi * 180)[0:-1] -
                                 (np.asarray(np.append(np.pi, rotated_angles)) / np.pi * 180)[1:])[
                                0:int(np.ceil(len(rotated_angles) / 2))]
                    distances = np.append(distances, distances[0:-1][::-1])
                    single = np.zeros_like(x)
                    for peak in range(0, len(angles_list[combo])):
                        mu = rotated_angles[peak] / np.pi * 180
                        variance = 1.5 * distances[peak]
                        sigma = math.sqrt(variance)
                        single += mlab.normpdf(x, mu, sigma) * rotated_results[peak]
                    beta += single
                    # plt.plot(x, single)

            # plt.show()
            # plt.savefig('combinaton_test_' + str(frame) + '.png')
            # plt.close()
            self.azimuth = np.argmax(alpha) / (len(alpha) / 180) / 180 * np.pi
            # alpha = np.argmax(alpha) / (len(alpha) / 180)
            self.elevation = np.argmax(beta) / (len(beta) / 180) / 180 * np.pi
            # beta = np.argmax(beta) / (len(beta) / 180)
            # print('DOA: ' + str(doa))
            # return [alpha, beta]

        def sph_2_cart(self, az, el, r):
            return r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

        def time_delay(self, speed_of_sound, mic, azimuth, elevation):
            mic_pair_vec = np.array([mic.x_loc, mic.y_loc, mic.z_loc])
            st_vec = self.sph_2_cart(azimuth, elevation, 1)
            delay = np.sum(mic_pair_vec * st_vec) / speed_of_sound
            return delay

    class MicInMatrix:
        def __init__(self, x, y, z):
            self.x_loc = x
            self.y_loc = y
            self.z_loc = z

    def __init__(self, n, f, speed_of_sound, mics_locs, frame_hop, frame_len, mu_cov):
        self.num_of_mics = n
        self.speed_of_sound = speed_of_sound
        self.frame_hop = frame_hop
        self.frame_len = frame_len
        self.frequency = f
        self.mu_cov = mu_cov
        self.psd_tracking_constant_speech = 0.975
        self.psd_tracking_constant_noise = 0.975
        self.frame = 0
        self.vad = None
        self.doa = None
        self.mics = [None] * self.num_of_mics
        self.fft_len = int(self.frame_len / 2 + 1)
        self.eigenvector = np.ones((self.fft_len, self.num_of_mics), dtype=np.float32)/self.num_of_mics
        self.psd_speech = np.random.rand((self.fft_len, self._num_of_mics, self.num_of_mics), dtype=np.float32)
        self.psd_noise = np.random.rand((self.fft_len, self._num_of_mics, self.num_of_mics), dtype=np.float32)
        self.spat_cov_mat = np.zeros((self.fft_len, self.num_of_mics, self.num_of_mics), dtype=np.complex64)
        for i in range(self.num_of_mics):
            self.mics[i] = self.MicInMatrix(mics_locs[i][0], mics_locs[i][1], mics_locs[i][2])
        self.angle_matrix = np.empty((self.num_of_mics, self.num_of_mics, 3), np.float32)
        self.distance_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.float32)
        self.max_delay_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.int)
        self.all_combs = list(itertools.combinations(range(n), 2))
        # self.all_combs = list(list(itertools.combinations(range(n), 2))[i] for i in [0, 1, 5, 12, 13, 14])
        self.angles_list = list()
        for comb in self.all_combs:
            self.distance_matrix[comb] = self.compute_dist(comb[0], comb[1])
            self.max_delay_matrix[comb] = int(self.calculate_max_delay(comb[0], comb[1]))
        self.calc_all_angles()

        # debugging variables
        self.vad_results = list()

    def estimate_covariance_mat(self, mask, signal):
        sig = signal.reshape(self.fft_len, 1, -1) @ np.conj(signal).reshape(self.fft_len, -1, 1)
        update = self.mu_cov * self.spat_cov_mat + (1 - self.mu_cov) * sig
        return (1 - mask) * self.spat_cov_mat + mask * update

    def fast_mvdr(self, sound, steervect):
        cminv = np.linalg.inv(self.spat_cov_mat)
        conj = np.conj(steervect).reshape(self.fft_len, 1, -1)
        return (conj @ cminv @ sound.reshape(self.fft_len, -1, 1)) / (
                conj @ cminv @ steervect.reshape(self.fft_len, -1, 1))

    def update_psds(self, fft_vector, speech_mask):
        # which PSDs will be updated
        toUpd = speech_mask == 1
        self.psd_speech[toUpd] = self.psd_tracking_constant_speech * self.psd_speech[toUpd] + \
                                 (1 - self.psd_tracking_constant_speech) * \
                                 np.dot(fft_vector[toUpd, :], fft_vector[toUpd, :].conj().T)

        toUpd = np.invert(toUpd)
        self.psd_noise[toUpd] = self.psd_tracking_constant_noise * self.psd_noise[toUpd] + \
                                (1 - self.psd_tracking_constant_noise) * \
                                np.dot(fft_vector[toUpd, :], fft_vector[toUpd, :].conj().T)

    def update_ev_by_power_iteration(self):
        unnormalized_eigenvector = np.einsum('...ij,...j->...i', self.psd_speech, self.eigenvector)
        self.eigenvector = unnormalized_eigenvector / \
                      np.sqrt(np.sum(np.abs(unnormalized_eigenvector)) ** 2 / self.num_of_mics)

    # compute angle between mics around axis
    def compute_ang(self, mic_1, mic_2):
        mic_vector = [self.mics[mic_2].x_loc - self.mics[mic_1].x_loc, self.mics[mic_2].y_loc - self.mics[mic_1].y_loc,
                      self.mics[mic_2].z_loc - self.mics[mic_1].z_loc]
        x_vec = [1, 0, 0]
        y_vec = [0, 1, 0]
        z_vec = [0, 0, 1]

        on_xy = np.arccos(((mic_vector[1] * x_vec[1]) +
                           (mic_vector[0] * x_vec[0])) /
                          ((np.sqrt(x_vec[1] ** 2 + x_vec[0] ** 2)) *
                           (np.sqrt(mic_vector[1] ** 2 + mic_vector[0] ** 2))))
        if np.isnan(on_xy):
            on_xy = 0.0

        on_yz = np.arccos(((mic_vector[1] * y_vec[1]) +
                           (mic_vector[2] * y_vec[2])) /
                          ((np.sqrt(y_vec[1] ** 2 + y_vec[2] ** 2)) *
                           (np.sqrt(mic_vector[1] ** 2 + mic_vector[2] ** 2))))
        if np.isnan(on_yz):
            on_yz = 0.0

        on_xz = np.arccos(((mic_vector[2] * z_vec[2]) +
                           (mic_vector[0] * z_vec[0])) /
                          ((np.sqrt(z_vec[2] ** 2 + z_vec[0] ** 2)) *
                           (np.sqrt(mic_vector[2] ** 2 + mic_vector[0] ** 2))))
        if np.isnan(on_xz):
            on_xz = 0.0

        return [on_xy, on_yz, on_xz]

    # compute distance between mics
    def compute_dist(self, mic_1, mic_2):
        return np.sqrt((self.mics[mic_2].x_loc - self.mics[mic_1].x_loc) ** 2 +
                       (self.mics[mic_2].y_loc - self.mics[mic_1].y_loc) ** 2 +
                       (self.mics[mic_2].z_loc - self.mics[mic_1].z_loc) ** 2)

    def calculate_max_delay(self, mic_1, mic_2):
        dt = 1 / self.frequency * self.speed_of_sound
        return np.floor(self.distance_matrix[mic_1, mic_2] / dt)

    def compute_angle(self, n, d):
        return np.arccos((1 / self.frequency * self.speed_of_sound * n) / d)

    def calc_all_angles(self):
        # dt = 1 / self.frequency * SPEED_OF_SOUND
        for comb in self.all_combs:
            angles = list()
            max_d = self.max_delay_matrix[comb[0], comb[1]]
            angle_indexes = list(range(-max_d, max_d + 1))
            for ang in angle_indexes:
                angles.append(self.compute_angle(ang, self.distance_matrix[comb[0], comb[1]]))
            self.angles_list.append(angles)

    def initialize(self):
        self.vad = self.VAD(mu_vad=0.975, init_est_vad=20, frame_len=self.frame_len, vad_thresh=4.8)
        self.doa = self.DOA()

    def process(self, ffts):
        vad_mask = self.vad.vad(ffts[:, 0])  # TODO change this to VAD mask

        self.spat_cov_mat = self.estimate_covariance_mat(vad_mask, ffts)

        self.update_psds(ffts, vad_mask)
        self.update_ev_by_power_iteration()

        # num_cores = multiprocessing.cpu_count()
        # results_array = Parallel(n_jobs=num_cores) \
        #     (delayed(self.doa.gcc_phat)(ffts[:, comb[0]], ffts[:, comb[1]],
        #                                 (self.max_delay_matrix[(comb[0], comb[1])] + 1))
        #      for comb in self.all_combs)

        # if vad_res <= self.vad.vad_thresh:
        #     self.doa.combine_gccs(self, self.angles_list, results_array, list(range(15)),
        #                           self.angle_matrix)

        # d_theta = np.ones((self.fft_len, self.num_of_mics), dtype=np.complex64)
        # factor_1 = -1j * 2 * np.pi
        # for k in range(1, self.fft_len):
        #     factor_2 = (k * self.frequency / (self.frame_len / 2))
        #     d_theta[k, :] = [1,
        #                      np.exp(factor_1 * self.doa.time_delay(self.speed_of_sound, self.mics[1], self.doa.azimuth,
        #                                                            self.doa.elevation) / factor_2),
        #                      np.exp(factor_1 * self.doa.time_delay(self.speed_of_sound, self.mics[2], self.doa.azimuth,
        #                                                            self.doa.elevation) / factor_2),
        #                      np.exp(factor_1 * self.doa.time_delay(self.speed_of_sound, self.mics[3], self.doa.azimuth,
        #                                                            self.doa.elevation) / factor_2),
        #                      np.exp(factor_1 * self.doa.time_delay(self.speed_of_sound, self.mics[4], self.doa.azimuth,
        #                                                            self.doa.elevation) / factor_2),
        #                      np.exp(factor_1 * self.doa.time_delay(self.speed_of_sound, self.mics[5], self.doa.azimuth,
        #                                                            self.doa.elevation) / factor_2)]

        result_fftd = self.fast_mvdr(ffts, self.eigenvector)

        return result_fftd.reshape(-1, 1)
