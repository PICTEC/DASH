import scipy.io.wavfile as sio
import scipy.signal as ss
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
# TODO: turn gcc into class

class matrix_class():
    def __init__(self, n, f):
        self.mics = [None] * n
        self.frequency = f
        self.angle_matrix = np.empty((n, n, 3), np.float32)
        self.distance_matrix = np.empty((n, n), np.float32)
        self.max_delay_matrix = np.empty((n, n), np.int)

    class mic_in_matrix():
        def __init__(self, x, y, z):
            self.x_loc = x
            self.y_loc = y
            self.z_loc = z

# compute angle between mics around axis
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


mat = matrix_class(N, 16000)
mat.mics[0] = mat.mic_in_matrix(0.00000001, 0.00000001, 0.00000001)
mat.mics[1] = mat.mic_in_matrix(0.1, 0.00000001, 0.00000001)
mat.mics[2] = mat.mic_in_matrix(0.2, 0.00000001, 0.00000001)
mat.mics[3] = mat.mic_in_matrix(0.00000001, -0.19, 0.00000001)
mat.mics[4] = mat.mic_in_matrix(0.1, -0.19, 0.00000001)
mat.mics[5] = mat.mic_in_matrix(0.2, -0.19, 0.00000001)
all_combs = list(itertools.combinations(range(N), 2))

for comb in all_combs:
        mat.distance_matrix[comb] = mat.compute_dist(comb[0], comb[1])
        mat.max_delay_matrix[comb] = int(mat.calculate_max_delay(comb[0], comb[1]))

global spec_avg
global frame
frame = 1

original_wav = sio.read('/home/kglowczewski/PICTEC/dash/dataset/0.wav')
original_wav = np.asarray(original_wav[1])
#original_wav = original_wav / 2**16

# x1 = np.linspace(-np.pi, 3*np.pi, 512)
# x2 = np.append(x1[10:512], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=0)


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
    sigl_fft = sfft.fft(sigl)
    sigr_fft = sfft.fft(sigr)
    sigr_fft_star = np.conj(sigr_fft)
    cc = sigl_fft*sigr_fft_star
    cc_phat = cc/abs(cc)
    r_phat = sfft.ifft(cc_phat)
    return r_phat

# n delay in samples from gcc
# fs sampling frequency
# c speed of sound
# d distance between mics


def compute_angle(c, n, d, fs=SAMPLING_FREQUENCY):
    return np.arccos(d/(fs*c*n))


def estimate_covariance_mat(signals_mat):
    cov_mat = np.zeros((N, N, int(signals_mat.shape[0]/2 + 1)), np.complex64)
    signals_mat_windowed = np.transpose(signals_mat) * np.hanning(signals_mat.shape[0])
    fftd = sfft.fft(signals_mat_windowed, axis=1)
    for k in range(0, int(fftd.shape[1]/2 + 1)):
        cov_mat[:, :, k] = np.outer(np.transpose(fftd[:, k]), np.conjugate(fftd[:, k]))
        cov_mat[:, :, k] = cov_mat[:, :, k] / np.trace(cov_mat[:, :, k])
    return cov_mat

# fig = plt.figure()
# plt.plot(gcc_phat(x1, x2))
# plt.show()

all_combs = list(itertools.combinations(range(N), 2))
vad_results = list()
for frame in range(int(np.floor(original_wav.shape[0]/FRAME_HOP) - 3)):
    results_array = list()
    vad_res = VAD(np.asarray(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), 0]))
    vad_results.append(vad_res)
    if frame == 0:
        spat_cov_mat = estimate_covariance_mat(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), :])
    if vad_res > VAD_THRESH:
        spat_cov_mat = spat_cov_mat * MU_COV + estimate_covariance_mat(original_wav[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN), :]) * (1 - MU_COV)

    for comb in all_combs:
        sig1 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[0]])
        sig2 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[1]])
        res = gcc_phat(sig1, sig2)[0:mat.max_delay_matrix[comb]]
        res = np.concatenate((res[::-1], res[1::]), axis=None)
        results_array.append(res)

        fig = plt.figure()
        plt.plot(res)
        # plt.show()
        plt.savefig('gcc_test_' + str(frame) + '_' + str(comb) + '.png')
        plt.close()

    result_fftd = np.zeros((sig1.shape[0]/2 + 1, N), np.complex64)
    for chan in range(N):
        result_fftd[:, chan] = sfft.fft(np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), chan]))

    if vad_res <= VAD_THRESH:
        for k in range(sig1.shape[0]/2 + 1):
            d_theta = np.zeros(N) # calculate steering vector based on combined gccs

            spat_cov_mat_inv = np.linalg.inv(spat_cov_mat[:, :, k])
            # TODO set proper multiplications below
            w_theta = (spat_cov_mat_inv * d_theta)/(np.conjugate(d_theta) * spat_cov_mat_inv * d_theta)
            result_fftd[k, :] = np.conjugate(w_theta) * result_fftd[k, :]

fig = plt.figure()
plt.plot(vad_results)
plt.show()
