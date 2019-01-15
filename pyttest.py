import scipy.io.wavfile as sio
import scipy.signal as ss
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt

FRAME_LEN = 512
FRAME_HOP = 256
SAMPLING_FREQUENCY = 16000
SPEED_OF_SOUND = 340
INIT_EST_VAD = 20
MU_VAD = 0.95
VAD_THRESH = 3
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
original_wav = original_wav / 2**16

x1 = np.linspace(-np.pi, 3*np.pi, 512)
x2 = np.append(x1[10:512], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=0)


def VAD(x, mu=MU_VAD, init_est=INIT_EST_VAD):
    global frame
    global spec_avg

    x_fft = sfft.rfft(x)
    x_fft += 0.01

    # TODO: change order to speed up a little
    if frame == 1:
        spec_avg = x_fft
        return -6
    elif frame < init_est:
        spec_avg += x_fft
        return -6
    elif frame == init_est:
        spec_avg /= init_est
        return -6
    else:
        whitened_spec = (x_fft/spec_avg) ^ 2
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

# n delay in samples from gcc
# fs sampling frequency
# c speed of sound
# d distance between mics


def compute_angle(c, n, d, fs=SAMPLING_FREQUENCY):
    return np.arccos(d/(fs*c*n))


# fig = plt.figure()
# plt.plot(gcc_phat(x1, x2))
# plt.show()

result_array = np.empty((N, N), None)
all_combs = list(itertools.combinations(range(N), 2))
for frame in range(int(np.floor(original_wav.shape[0]/FRAME_HOP) - 1)):
    for comb in all_combs:
        sig1 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[0]])
        sig2 = np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), comb[1]])
        result_array[comb] = np.concatenate(gcc_phat(sig1, sig2)[0:mat.max_delay_matrix[comb]],
                                            gcc_phat(sig2, sig1)[0:mat.max_delay_matrix[comb]])

    #fig = plt.figure()

    #plt.plot(gcc_phat(sig2, sig1)[0:10])
    # plt.show()
    #plt.savefig('gcc_test_' + str(frame) + '.png')
    #plt.close()
