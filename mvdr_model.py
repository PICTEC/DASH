import scipy.io.wavfile as sio
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

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
            self.spec_avg = np.ones(int(frame_len/2 + 1))

        def vad(self, fftd):
            fftd += 0.00000001
            whitened_spec = (fftd / self.spec_avg) ** 2
            self.spec_avg = self.spec_avg * self.mu_vad + fftd * (1 - self.mu_vad)
            p = whitened_spec / np.sum(whitened_spec)
            return -np.sum(p * np.log(p))

    class DOA:
        def __init__(self):
            pass

        def gcc_phat(self, sigl_fft, sigr_fft):
            sigr_fft_star = np.conj(sigr_fft)
            cc = sigl_fft * sigr_fft_star
            cc_phat = cc / abs(cc)
            r_phat = sfft.irfft(cc_phat)
            return r_phat

        def combine_gccs(self, angles_list, results_array, combs_list):
            x = np.linspace(0, 180, 1080)
            y = np.zeros_like(x)
            for combo in combs_list:
                distances = ((np.asarray(np.append(np.pi, angles_list[combo])) / np.pi * 180)[0:-1] -
                             (np.asarray(np.append(np.pi, angles_list[combo])) / np.pi * 180)[1:])[
                            0:int(np.ceil(len(angles_list[combo]) / 2))]
                distances = np.append(distances, distances[0:-1][::-1])

                for peak in range(0, len(angles_list[combo])):
                    mu = angles_list[combo][peak] / np.pi * 180
                    variance = 1.5 * distances[peak]
                    sigma = math.sqrt(variance)
                    y += mlab.normpdf(x, mu, sigma) * results_array[combo][peak]
            # plt.plot(x, y)
            # plt.show()
            # plt.savefig('combinaton_test_' + str(frame) + '.png')
            # plt.close()
            doa = np.argmax(y) / (len(y) / 180)
            return doa

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
        self.frame = 0
        self.vad = None
        self.doa = None
        self.mics = [None] * self.num_of_mics
        self.fft_len = int(self.frame_len / 2 + 1)
        self.spat_cov_mat = np.zeros((self.num_of_mics, self.num_of_mics, self.fft_len), np.complex64)
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

    def estimate_covariance_mat(self, fftd):
        cov_mat = np.zeros((self.num_of_mics, self.num_of_mics, int(self.fft_len)), np.complex64)
        for k in range(0, self.fft_len):
            cov_mat[:, :, k] = np.outer(np.transpose(fftd[k, :]), np.conjugate(fftd[k, :]))
            cov_mat[:, :, k] = cov_mat[:, :, k] / np.trace(cov_mat[:, :, k])
        return cov_mat

# compute angle between mics around axis # is this even necessary?
    def on_x(self, mic_1, mic_2):
        return np.arccos(((self.mics[mic_2].y_loc * self.mics[mic_1].y_loc) +
                          (self.mics[mic_2].z_loc * self.mics[mic_1].z_loc)) /
                         ((np.sqrt(self.mics[mic_1].y_loc**2 + self.mics[mic_1].z_loc**2)) *
                          (np.sqrt(self.mics[mic_2].y_loc**2 + self.mics[mic_2].z_loc**2))))

    def on_y(self, mic_1, mic_2):
        return np.arccos(((self.mics[mic_2].x_loc * self.mics[mic_1].x_loc) +
                          (self.mics[mic_2].z_loc * self.mics[mic_1].z_loc)) /
                         ((np.sqrt(self.mics[mic_1].x_loc**2 + self.mics[mic_1].z_loc**2)) *
                          (np.sqrt(self.mics[mic_2].x_loc**2 + self.mics[mic_2].z_loc**2))))

    def on_z(self, mic_1, mic_2):
        return np.arccos(((self.mics[mic_2].y_loc * self.mics[mic_1].y_loc) +
                          (self.mics[mic_2].x_loc * self.mics[mic_1].x_loc)) /
                         ((np.sqrt(self.mics[mic_1].y_loc**2 + self.mics[mic_1].x_loc**2)) *
                          (np.sqrt(self.mics[mic_2].y_loc**2 + self.mics[mic_2].x_loc**2))))

    def compute_ang(self, mic_1, mic_2, axis):
        if axis == 'x':
            ang = self.on_x(mic_1, mic_2)
        elif axis == 'y':
            ang = self.on_y(mic_1, mic_2)
        elif axis == 'z':
            ang = self.on_z(mic_1, mic_2)
        return ang

# compute distance between mics
    def compute_dist(self, mic_1, mic_2):
        return np.sqrt((self.mics[mic_2].x_loc - self.mics[mic_1].x_loc)**2 +
                       (self.mics[mic_2].y_loc - self.mics[mic_1].y_loc)**2 +
                       (self.mics[mic_2].z_loc - self.mics[mic_1].z_loc)**2)

    def calculate_max_delay(self, mic_1, mic_2):
        dt = 1/self.frequency * self.speed_of_sound
        return np.floor(self.distance_matrix[mic_1, mic_2]/dt)

    # n delay in samples from gcc
    # c speed of sound
    # d distance between mics

    def compute_angle(self, n, d):
        return np.arccos((1/self.frequency * self.speed_of_sound * n) / d)

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
        results_array = list()
        vad_res = self.vad.vad(ffts[:, 0])  # TODO check if this is ok
        self.vad_results.append(vad_res)

        if vad_res > self.vad.vad_thresh or self.frame < 10:
            self.spat_cov_mat = self.spat_cov_mat * self.mu_cov + self.estimate_covariance_mat(
                ffts) * (1 - self.mu_cov)

        for comb in self.all_combs:
            res = self.doa.gcc_phat(ffts[:, comb[0]], ffts[:, comb[1]])[0:(self.max_delay_matrix[comb] + 1)]
            res = np.concatenate((res[::-1], res[1::]), axis=None)
            results_array.append(res)

            # fig = plt.figure()
            # plt.plot(res)
            # plt.show()
            # plt.savefig('gcc_test_' + str(frame) + '_' + str(comb) + '.png')
            # plt.close()

        if vad_res <= self.vad.vad_thresh:
            doa_res = self.doa.combine_gccs(self.angles_list, results_array, [0, 1, 5, 12, 13, 14]) / 180 * np.pi

        result_fftd = np.zeros((self.fft_len, self.num_of_mics), np.complex64)

        # if vad_res <= VAD_THRESH: # do we replace it with something?
        for k in range(1, self.fft_len):
            # this is VERY specific to current implementation, change it if combine_gccs changes
            d_theta = [1,
                       np.exp(-1j * 2 * np.pi * 0.1 / (k * self.frequency / (self.frame_len / 2)) * np.cos(doa_res)),
                       np.exp(-1j * 2 * np.pi * 0.2 / (k * self.frequency / (self.frame_len/ 2)) * np.cos(doa_res)),
                       1,
                       np.exp(-1j * 2 * np.pi * 0.1 / (k * self.frequency / (self.frame_len / 2)) * np.cos(doa_res)),
                       np.exp(-1j * 2 * np.pi * 0.2 / (k * self.frequency / (self.frame_len / 2)) * np.cos(doa_res))]
            # d_theta = np.zeros(mat.mic)

            spat_cov_mat_inv = np.linalg.inv(self.spat_cov_mat[:, :, k])
            # this should be right
            w_theta = np.matmul(np.conjugate(d_theta), spat_cov_mat_inv) / np.matmul(
                np.matmul(np.conjugate(d_theta), spat_cov_mat_inv), d_theta)
            result_fftd[k, :] = w_theta * ffts[k, :]

        sig_summed = np.sum(result_fftd, axis=1)
        sig_summed = ffts[:,0]
        self.frame += 1
        print(self.frame)

        return sig_summed
