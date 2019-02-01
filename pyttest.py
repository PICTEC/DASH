import scipy.io.wavfile as sio
import scipy.signal as ss
import scipy.fftpack as sfft
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import sounddevice as sd
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
        self.angle_matrix = np.zeros((self.num_of_mics, self.num_of_mics, 3), np.float32)
        self.distance_matrix = np.zeros((self.num_of_mics, self.num_of_mics), np.float32)
        self.max_delay_matrix = np.zeros((self.num_of_mics, self.num_of_mics), np.int)
        self.all_combs = list(itertools.combinations(range(n), 2))
        self.angles_list = list()
        for comb in self.all_combs:
            self.distance_matrix[comb] = self.compute_dist(comb[0], comb[1])
            self.max_delay_matrix[comb] = int(self.calculate_max_delay(comb[0], comb[1]))
            self.angle_matrix[comb[0], comb[1]] = self.compute_ang(comb[0], comb[1])

        self.calc_all_angles()


    class mic_in_matrix():
        def __init__(self, x, y, z):
            self.x_loc = x
            self.y_loc = y
            self.z_loc = z

# compute angle between mics around axis # is this even necessary?
#     def on_x(self, mic_1, mic_2):
#         return np.arccos(((self.mics[mic_2].y_loc * self.mics[mic_1].y_loc) +
#                           (self.mics[mic_2].z_loc * self.mics[mic_1].z_loc)) /
#                          ((np.sqrt(self.mics[mic_1].y_loc**2 + self.mics[mic_1].z_loc**2)) *
#                           (np.sqrt(self.mics[mic_2].y_loc**2 + self.mics[mic_2].z_loc**2))))
#
#     def on_y(self, mic_1, mic_2):
#         return np.arccos(((self.mics[mic_2].x_loc * self.mics[mic_1].x_loc) +
#                           (self.mics[mic_2].z_loc * self.mics[mic_1].z_loc)) /
#                          ((np.sqrt(self.mics[mic_1].x_loc**2 + self.mics[mic_1].z_loc**2)) *
#                           (np.sqrt(self.mics[mic_2].x_loc**2 + self.mics[mic_2].z_loc**2))))
#
#     def on_z(self, mic_1, mic_2):
#         return np.arccos(((self.mics[mic_2].y_loc * self.mics[mic_1].y_loc) +
#                           (self.mics[mic_2].x_loc * self.mics[mic_1].x_loc)) /
#                          ((np.sqrt(self.mics[mic_1].y_loc**2 + self.mics[mic_1].x_loc**2)) *
#                           (np.sqrt(self.mics[mic_2].y_loc**2 + self.mics[mic_2].x_loc**2))))

    def compute_ang(self, mic_1, mic_2):
        mic_vector = [self.mics[mic_2].x_loc - self.mics[mic_1].x_loc, self.mics[mic_2].y_loc - self.mics[mic_1].y_loc,
                      self.mics[mic_2].z_loc - self.mics[mic_1].z_loc]
        x_vec = [1, 0, 0]
        y_vec = [0, 1, 0]
        z_vec = [0, 0, 1]

        on_xy = np.arccos(((mic_vector[1] * x_vec[1]) +
                           (mic_vector[0] * x_vec[0])) /
                          ((np.sqrt(x_vec[1]**2 + x_vec[0]**2)) *
                           (np.sqrt(mic_vector[1]**2 + mic_vector[0]**2))))
        if np.isnan(on_xy):
            on_xy = 0.0

        on_yz = np.arccos(((mic_vector[1] * y_vec[1]) +
                           (mic_vector[2] * y_vec[2])) /
                          ((np.sqrt(y_vec[1]**2 + y_vec[2]**2)) *
                           (np.sqrt(mic_vector[1]**2 + mic_vector[2]**2))))
        if np.isnan(on_yz):
            on_yz = 0.0

        on_xz = np.arccos(((mic_vector[2] * z_vec[2]) +
                           (mic_vector[0] * z_vec[0])) /
                          ((np.sqrt(z_vec[2] ** 2 + z_vec[0] ** 2)) *
                           (np.sqrt(mic_vector[2] ** 2 + mic_vector[0] ** 2))))
        if np.isnan(on_xz):
            on_xz = 0.0

        return [on_xy, on_yz, on_xz]


        # if axis == 'x':
        #     ang = self.on_x(mic_1, mic_2)
        # elif axis == 'y':
        #     ang = self.on_y(mic_1, mic_2)
        # elif axis == 'z':
        #     ang = self.on_z(mic_1, mic_2)
        # return ang

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
                   0, 0, 0,
                   0.1, 0, 0,
                   0.2, 0, 0,
                   0, -0.19, 0,
                   0.1, -0.19, 0,
                   0.2, -0.19, 0)


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


def sph_2_cart(az, el, r):
    return r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def time_delay(speed_of_sound, mic, azimuth, elevation):
    mic_pair_vec = np.array([mic.x_loc, mic.y_loc, mic.z_loc])
    st_vec = sph_2_cart(azimuth, elevation, 1)
    delay = np.sum(mic_pair_vec * st_vec) / speed_of_sound
    return delay


def combine_gccs(angles_list, results_array, combs_list, angle_matrix):
    x = np.linspace(0, 180, 1080)
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
                         (np.asarray(np.append(np.pi, rotated_angles)) / np.pi * 180)[1:])[0:int(np.ceil(len(rotated_angles)/2))]
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
    alpha = np.argmax(alpha) / (len(alpha)/180)
    beta = np.argmax(beta) / (len(beta) / 180)
    # print('DOA: ' + str(doa))
    return [alpha, beta]
# fig = plt.figure()
# plt.plot(gcc_phat(x1, x2))
# plt.show()

# all_combs = list(itertools.combinations(range(N), 2))
vad_results = list()
DOA_az = np.pi/4
DOA_el = np.pi/4
output = np.zeros(original_wav.shape[0], np.float)
rise = np.arange(0, int(FRAME_LEN/2 - FRAME_HOP/2), 1)
fall = rise[::-1]
doas = np.zeros(int(np.floor(original_wav.shape[0]/FRAME_HOP) - 3))
reconstructing_window = np.concatenate((rise, np.repeat(191, 128), fall))
reconstructing_window = reconstructing_window/np.max(reconstructing_window)
for frame in range(int(np.floor(original_wav.shape[0]/FRAME_HOP) - 3)):
    print('frame: ' + str(frame))
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
        DOA_az, DOA_el = np.asarray(combine_gccs(mat.angles_list, results_array, list(range(15)), mat.angle_matrix)) / 180 * np.pi
    # doas[frame] = DOA*180/np.pi
    result_fftd = np.zeros((int(sig1.shape[0]/2 + 1), N), np.complex64)
    for chan in range(N):
        result_fftd[:, chan] = sfft.fft(np.asarray(original_wav[(frame*FRAME_HOP):(frame*FRAME_HOP + FRAME_LEN), chan]))[0:int(sig1.shape[0]/2 + 1)]

    # if vad_res <= VAD_THRESH: # do we replace it with something?
    for k in range(1, int(sig1.shape[0]/2 + 1)):
        # this is VERY specific to current implementation, change it if combine_gccs changes
        d_theta = [1,
                   np.exp(-1j * 2 * np.pi * time_delay(340, mat.mics[1], DOA_az, DOA_el) / (k * mat.frequency / (FRAME_LEN / 2))),
                   np.exp(-1j * 2 * np.pi * time_delay(340, mat.mics[2], DOA_az, DOA_el) / (k * mat.frequency / (FRAME_LEN / 2))),
                   np.exp(-1j * 2 * np.pi * time_delay(340, mat.mics[3], DOA_az, DOA_el) / (k * mat.frequency / (FRAME_LEN / 2))),
                   np.exp(-1j * 2 * np.pi * time_delay(340, mat.mics[4], DOA_az, DOA_el) / (k * mat.frequency / (FRAME_LEN / 2))),
                   np.exp(-1j * 2 * np.pi * time_delay(340, mat.mics[5], DOA_az, DOA_el) / (k * mat.frequency / (FRAME_LEN / 2)))]
            # d_theta = np.zeros(mat.mic)

        spat_cov_mat_inv = np.linalg.inv(spat_cov_mat[:, :, k])
        # this should be right
        w_theta = np.matmul(np.conjugate(d_theta), spat_cov_mat_inv)/np.matmul(
            np.matmul(np.conjugate(d_theta), spat_cov_mat_inv), d_theta)
        result_fftd[k, :] = w_theta * result_fftd[k, :]

    sig_summed = np.sum(result_fftd, axis=1)
    sig_out = sfft.ifft(np.concatenate((sig_summed[::-1], sig_summed[1:(sig_summed.shape[0] - 1)])))
    output[(frame * FRAME_HOP):(frame * FRAME_HOP + FRAME_LEN)] = sig_out * reconstructing_window

scaled = np.int16(output/np.max(np.abs(output)) * 32767)
sd.play(scaled, 16000)
sio.write('test.wav', 16000, scaled)

# fig = plt.figure()
# plt.plot(vad_results)
# plt.show()
