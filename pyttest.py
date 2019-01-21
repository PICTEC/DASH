import scipy.io.wavfile as sio
import scipy.signal as ss
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import time

FRAME_LEN = 512
FRAME_HOP = 128
SAMPLING_FREQUENCY = 16000
SPEED_OF_SOUND = 340
INIT_EST_VAD = 25
MU_VAD = 0.975
MU_COV = 0.95
VAD_THRESH = 4.8
N = 6

# TODO: turn VAD into class
# TODO: add gcc into matrix_class


class matrix_class():
    def __init__(self, n, f, *argv):
        self.num_of_mics = n
        self.mics = [None] * self.num_of_mics
        for i in range(self.num_of_mics):
            self.mics[i] = self.mic_in_matrix(argv[3*i], argv[3*i + 1], argv[3*i + 2])
        self.frequency = f
        self.angle_matrix = np.empty((self.num_of_mics, self.num_of_mics, 3), np.float32)
        self.distance_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.float32)
        self.max_delay_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.int)
        self.all_combs = list(itertools.combinations(range(n), 2))
        self.angles_list = list()
        for comb in self.all_combs:
            self.distance_matrix[comb] = self.compute_dist(comb[0], comb[1])
            self.max_delay_matrix[comb] = int(self.calculate_max_delay(comb[0], comb[1]))
        self.calc_all_angles()

    class mic_in_matrix():
        def __init__(self, x, y, z):
            self.x_loc = x
            self.y_loc = y
            self.z_loc = z

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
        dt = 1/self.frequency * SPEED_OF_SOUND
        return np.floor(self.distance_matrix[mic_1, mic_2]/dt)

    # n delay in samples from gcc
    # c speed of sound
    # d distance between mics

    def compute_angle(self, c, n, d):
        return np.arccos((1/self.frequency * c * n) / d)

    def calc_all_angles(self):
        # dt = 1 / self.frequency * SPEED_OF_SOUND
        for comb in self.all_combs:
            angles = list()
            max_d = self.max_delay_matrix[comb[0], comb[1]]
            angle_indexes = list(range(-max_d, max_d + 1))
            for ang in angle_indexes:
                angles.append(self.compute_angle(340, ang, self.distance_matrix[comb[0], comb[1]]))
            self.angles_list.append(angles)


# this defines matrix # 0.00000001 because 0 causes problems with /0
mat = matrix_class(6, 16000,
                   0.00000001, 0.00000001, 0.00000001,
                   0.1, 0.00000001, 0.00000001,
                   0.2, 0.00000001, 0.00000001,
                   0.00000001, -0.19, 0.00000001,
                   0.1, -0.19, 0.00000001,
                   0.2, -0.19, 0.00000001)


global spec_avg
global frame
frame = 1

original_wav = sio.read('/home/kglowczewski/PICTEC/dash/dataset/0.wav')
original_wav = np.asarray(original_wav[1])
#original_wav = original_wav / 2**16

#x1 = np.linspace(-np.pi, 3*np.pi, 512)
#x2 = np.append(x1[10:512], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=0)


def VAD(x, mu=MU_VAD, init_est=INIT_EST_VAD):
    global frame
    global spec_avg

    x_fft = np.abs(sfft.fft(x * np.hanning(len(x))) / len(x))[0:int(len(x)/2 + 1)]
    x_fft += 0.0001

    # TODO: change order to speed up a little
    if frame == 0:
        spec_avg = x_fft
        return 5
    elif frame < init_est:
        spec_avg += x_fft
        return 5
    elif frame == init_est:
        spec_avg /= init_est
        return 5
    else:
        whitened_spec = (x_fft/spec_avg) ** 2
        spec_avg = spec_avg * mu + x_fft * (1 - mu)
        p = whitened_spec/np.sum(whitened_spec)
        return -np.sum(p*np.log(p))


def gcc_phat(sigl, sigr, len=FRAME_LEN, fs=SAMPLING_FREQUENCY):
    sigl_fft = sfft.rfft(sigl)
    sigr_fft = sfft.rfft(sigr)
    sigr_fft_star = np.conj(sigr_fft)
    cc = sigl_fft*sigr_fft_star
    cc_phat = cc/abs(cc)
    r_phat = sfft.irfft(cc_phat)
    return r_phat


def estimate_covariance_mat(signals_mat):
    cov_mat = np.zeros((N, N, int(signals_mat.shape[0]/2 + 1)), np.complex64)
    signals_mat_windowed = np.transpose(signals_mat) * np.hanning(signals_mat.shape[0])
    fftd = sfft.fft(signals_mat_windowed, axis=1)
    for k in range(0, int(fftd.shape[1]/2 + 1)):
        cov_mat[:, :, k] = np.outer(np.transpose(fftd[:, k]), np.conjugate(fftd[:, k]))
        cov_mat[:, :, k] = cov_mat[:, :, k] / np.trace(cov_mat[:, :, k])
    return cov_mat


# TODO make it so it uses ALL microphone pairs
def combine_gccs(angles_list, results_array, combs_list):
    x = np.linspace(0, 180, 1080)
    y = np.zeros_like(x)
    for combo in combs_list:
        distances = ((np.asarray(np.append(np.pi, angles_list[combo])) / np.pi * 180)[0:-1] -
                     (np.asarray(np.append(np.pi, angles_list[combo])) / np.pi * 180)[1:])[0:int(np.ceil(len(angles_list[combo])/2))]
        distances = np.append(distances, distances[0:-1][::-1])
        for peak in range(0, len(angles_list[combo])):
            mu = angles_list[combo][peak] / np.pi * 180
            variance = 1.5 * distances[peak]
            sigma = math.sqrt(variance)
            y += mlab.normpdf(x, mu, sigma) * results_array[combo][peak]

    plt.plot(x, y)
    plt.show()
    plt.savefig('combinaton_test_' + str(frame) + '.png')
    plt.close()
    doa = np.argmax(y)/(len(y)/180)
    return doa
# fig = plt.figure()
# plt.plot(gcc_phat(x1, x2))
# plt.show()


# all_combs = list(itertools.combinations(range(N), 2))
vad_results = list()
DOA = 90
for frame in range(int(np.floor(original_wav.shape[0]/FRAME_HOP) - 3)):
    results_array = list()
    vad_res = VAD(np.asarray(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), 0]))
    vad_results.append(vad_res)
    if frame == 0:
        spat_cov_mat = estimate_covariance_mat(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), :])
    if vad_res > VAD_THRESH:
        spat_cov_mat = spat_cov_mat * MU_COV + estimate_covariance_mat(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), :]) * (1 - MU_COV)

    for comb in mat.all_combs:
        sig1 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[0]])
        sig2 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[1]])
        res = gcc_phat(sig1, sig2)[0:mat.max_delay_matrix[comb] + 1]
        res = np.concatenate((res[::-1], res[1::]), axis=None)
        results_array.append(res)

        # fig = plt.figure()
        # plt.plot(res)
        # plt.show()
        # plt.savefig('gcc_test_' + str(frame) + '_' + str(comb) + '.png')
        # plt.close()

    if vad_res <= VAD_THRESH:
        DOA = combine_gccs(mat.angles_list, results_array, [0, 1, 5, 12, 13, 14]) / 180 * np.pi

    result_fftd = np.zeros((int(sig1.shape[0]/2 + 1), N), np.complex64)
    for chan in range(N):
        result_fftd[:, chan] = sfft.fft(np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), chan]))[0:int(sig1.shape[0]/2 + 1)]

    # if vad_res <= VAD_THRESH: # do we replace it with something?
        for k in range(1, int(sig1.shape[0]/2 + 1)):
            # this is VERY specific to current implementation, change it if combine_gccs changes
            d_theta = [1,
                       np.exp(-1j * 2 * np.pi * 0.1 / (k * mat.frequency / (FRAME_LEN / 2)) * np.cos(DOA)),
                       np.exp(-1j * 2 * np.pi * 0.2 / (k * mat.frequency / (FRAME_LEN / 2)) * np.cos(DOA)),
                       1,
                       np.exp(-1j * 2 * np.pi * 0.1 / (k * mat.frequency / (FRAME_LEN / 2)) * np.cos(DOA)),
                       np.exp(-1j * 2 * np.pi * 0.2 / (k * mat.frequency / (FRAME_LEN / 2)) * np.cos(DOA))]
            # d_theta = np.zeros(mat.mic)

            spat_cov_mat_inv = np.linalg.inv(spat_cov_mat[:, :, k])
            # this should be right
            w_theta = np.matmul(spat_cov_mat_inv, d_theta)/np.matmul(
                np.matmul(np.conjugate(d_theta), spat_cov_mat_inv), d_theta)
            result_fftd[k, :] = np.conjugate(w_theta) * result_fftd[k, :]

    sig_summed = np.sum(result_fftd, axis=1)
    sig_out = sfft.ifft(np.concatenate(sig_summed[::-1], sig_summed[1::]))

fig = plt.figure()
plt.plot(vad_results)
plt.show()
