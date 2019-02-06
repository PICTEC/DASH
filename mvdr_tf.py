import itertools
import keras.backend as K
import numpy as np
import tensorflow as tf

def vad(fftd, spec_avg, mu_vad=0.95):
    """
    Voice activity detection
    Parameters: mu_vad
    Inputs: fftd
    State: spec_avg
    (fftd, spec_avg_bis) -> (out, spec_avg_bis)
    """
    fftd += 1e-8   # some epsilon for numeric stability
    whitened_spec = (fftd / spec_avg) ** 2
    spec_avg_bis = spec_avg * mu_vad + fftd * (1 - mu_vad)
    p = whitened_spec / K.sum(whitened_spec)
    return -K.sum(p * K.log(p)), spec_avg_bis

def doa_gcc_phat(sigl_fft, sigr_fft):
    """
    Params
    """
    sigr_fft_star = tf.conj(sigr_fft)
    cc = sigl_fft * sigr_fft_star
    cc_phat = cc / K.abs(cc)
    r_phat = tf.spectral.irfft(cc_phat)
    return r_phat

def sph_2_cart(az, el, r):
    return r * K.stack([K.cos(el) * K.cos(az), K.cos(el) * K.sin(az), K.sin(el)])

def time_delay(speed_of_sound, mic_array, azimuth, elevation):
    st_vec = sph_2_cart(azimuth, elevation, 1.0)
    return K.sum(mic_array * st_vec) / speed_of_sound

def compute_dist(mics, mic_1, mic_2):  # bind to mics
    return np.sqrt((mics[mic_2].x_loc - mics[mic_1].x_loc)**2 +
                   (mics[mic_2].y_loc - mics[mic_1].y_loc)**2 +
                   (mics[mic_2].z_loc - mics[mic_1].z_loc)**2)

def estimate_covariance_mat(fftd):
    a = K.expand_dims(fftd, axis=2)
    b = tf.conj(K.expand_dims(fftd, axis=1))
    cov_mat = tf.matmul(a, b)
    cov_mat /= K.expand_dims(K.expand_dims(tf.trace(cov_mat), -1), -1)
    return cov_mat

def combine_gccs(self, mat, angles_list, results_array, combs_list, angle_matrix):
    x = tf.linspace(0., 180., 1080)
    alpha = tf.zeros_like(x)
    beta = tf.zeros_like(x)
    for combo in combs_list:
        full_combo = mat.all_combs[combo]
        if np.round(mat.angle_matrix[full_combo[0], full_combo[1], 0] * 180 / np.pi) != 90:
            # this is static
            rotated_angles = angles_list[combo] - angle_matrix[mat.all_combs[combo]][0]
            rotated_angles_2 = np.append(rotated_angles[np.argwhere(rotated_angles < 0)] + np.pi,
                                       rotated_angles[np.argwhere(rotated_angles > 0)])
            distances = ((np.asarray(np.append(np.pi, rotated_angles_2)) / np.pi * 180)[0:-1] -
                         (np.asarray(np.append(np.pi, rotated_angles_2)) / np.pi * 180)[1:])[
                        0:int(np.ceil(len(rotated_angles_2) / 2))]
            distances = np.append(distances, distances[0:-1][::-1])
            # this is dynamic
            rotated_results = np.append(results_array[combo][np.argwhere(rotated_angles < 0)],
                                        results_array[combo][np.argwhere(rotated_angles > 0)])
            single = np.zeros_like(x)
            for peak in range(0, len(angles_list[combo])):
                mu = rotated_angles_2[peak] / np.pi * 180
                variance = 1.5 * distances[peak]
                sigma = math.sqrt(variance)
                single += mlab.normpdf(x, mu, sigma) * rotated_results[peak]
            alpha += single
        if np.round(mat.angle_matrix[full_combo[0], full_combo[1], 0] * 180 / np.pi) != 0:
            # this is static
            rotated_angles = angles_list[combo] - angle_matrix[mat.all_combs[combo]][0] - np.pi / 2
            rotated_angles_2 = np.append(rotated_angles[np.argwhere(rotated_angles < 0)] + np.pi,
                                       rotated_angles[np.argwhere(rotated_angles > 0)])
            distances = ((np.asarray(np.append(np.pi, rotated_angles_2)) / np.pi * 180)[0:-1] -
                         (np.asarray(np.append(np.pi, rotated_angles_2)) / np.pi * 180)[1:])[
                        0:int(np.ceil(len(rotated_angles_2) / 2))]
            distances = np.append(distances, distances[0:-1][::-1])
            single = np.zeros_like(x)
            # this is dynamic
            rotated_results = np.append(results_array[combo][np.argwhere(rotated_angles < 0)],
                                        results_array[combo][np.argwhere(rotated_angles > 0)])
            for peak in range(0, len(angles_list[combo])):
                mu = rotated_angles_2[peak] / np.pi * 180
                variance = 1.5 * distances[peak]
                sigma = math.sqrt(variance)
                single += mlab.normpdf(x, mu, sigma) * rotated_results[peak]
            beta += single
    return alpha, beta


class MicInMatrix:
    def __init__(self, x, y, z):
        self.x_loc = x
        self.y_loc = y
        self.z_loc = z


class Model:

    def __init__(self, n, f, speed_of_sound, mics_locs, frame_hop, frame_len, mu_cov):
        self.num_of_mics = n
        self.speed_of_sound = speed_of_sound
        self.frame_hop = frame_hop
        self.frame_len = frame_len
        self.frequency = f
        self.mu_cov = mu_cov
        self.mics = [None] * self.num_of_mics
        self.fft_len = int(self.frame_len / 2 + 1)
        self.spat_cov_mat = np.zeros((self.num_of_mics, self.num_of_mics, self.fft_len), np.complex64)
        for i in range(self.num_of_mics):
            self.mics[i] = MicInMatrix(mics_locs[i][0], mics_locs[i][1], mics_locs[i][2])
        self.angle_matrix = np.empty((self.num_of_mics, self.num_of_mics, 3), np.float32)
        self.distance_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.float32)
        self.max_delay_matrix = np.empty((self.num_of_mics, self.num_of_mics), np.int)
        self.all_combs = list(itertools.combinations(range(n), 2))
        self.angles_list = list()
        self.vad_thresh = 4.8
        for comb in self.all_combs:
            self.distance_matrix[comb] = compute_dist(self.mics, comb[0], comb[1])
            self.max_delay_matrix[comb] = int(self.calculate_max_delay(comb[0], comb[1]))

    def calculate_max_delay(self, mic_1, mic_2):
        dt = 1 / self.frequency * self.speed_of_sound
        return np.floor(self.distance_matrix[mic_1, mic_2] / dt)

    def compute_angle(self, n, d):
        return np.arccos((1/self.frequency * self.speed_of_sound * n) / d)

    def calc_all_angles(self):
        for comb in self.all_combs:
            angles = list()
            max_d = self.max_delay_matrix[comb[0], comb[1]]
            angle_indexes = list(range(-max_d, max_d + 1))
            for ang in angle_indexes:
                angles.append(self.compute_angle(ang, self.distance_matrix[comb[0], comb[1]]))
            self.angles_list.append(angles)

    def update_covmat_if_vad(self,
                             ffts,
                             spat_cov_mat,
                             frame,
                             spec_avg
                             ):
        vad_res, spec_avg_bis = vad(ffts[:, 0], spec_avg)  # self.vad
        spat_cov_mat = K.switch(
            tf.math.logical_or(vad_res > self.vad_thresh, frame < 10),
            spat_cov_mat * self.mu_cov + estimate_covariance_mat(ffts) * (1 - self.mu_cov),
            spat_cov_mat
        )
        return spat_cov_mat, vad_res, spec_avg_bis


    def program(self):
        print("Building Tensorflow program...")
        ### Definitions of tensors to be fed
        ffts = tf.placeholder(tf.float32, shape=(self.fft_len, self.num_of_mics))
        spat_cov_mat = tf.placeholder(tf.float32, shape=(self.fft_len, self.num_of_mics, self.num_of_mics))
        frame = tf.placeholder(tf.int64, shape=())
        spec_avg = tf.placeholder(tf.float32, shape=(self.fft_len))
        azimuth = tf.placeholder(tf.float32, shape=())
        elevation = tf.placeholder(tf.float32, shape=())
        ### Program itself
        # update covariance matrix (and return VAD estimation)
        spat_cov_mat, vad_res, spec_avg_bis = self.update_covmat_if_vad(ffts, spat_cov_mat, frame, spec_avg)
        # calculate GCC_PHAT constituents for each microphone pair
        results_array = []
        for comb in self.all_combs:
            res = doa_gcc_phat(ffts[:, comb[0]], ffts[:, comb[1]])[0:(self.max_delay_matrix[comb] + 1)]
            res = K.concatenate((res[::-1], res[1::]), axis=None)
            results_array.append(res)
        # combine all GCC elements and update azimuth and elevation if VAD
        alpha, beta = combine_gccs(self, self.angles_list, results_array, list(range(15)),
                                   self.angle_matrix)  # TODO: review this call
        azimuth = K.switch(
            vad_res <= self.vad_thresh,
            tf.argmax(alpha) / (len(alpha) / 180) / 180 * tf.pi,
            azimuth
        )
        elevation = K.switch(
            vad_res <= self.vad_thresh,
            tf.argmax(beta) / (len(beta) / 180) / 180 * tf.pi,
            elevation
        )
        # MVDR in the chosen angle
        spat_cov_mat_inv = tf.linalg.inv(spat_cov_mat)  # does axis agree (it was [:,:,k])
        d_theta = [K.zeros((self.fft_len - 1),)] + [
            K.exp(-1j * 2 * tf.pi *
                self.doa.time_delay(self.speed_of_sound, mic, azimuth, elevation)
                       (K.arange(1, self.fft_len) * self.frequency / (self.frame_len / 2))
            ) for mic in self.mics[1:]
        ]
        w_theta = K.dot(tf.conj(d_theta), spat_cov_mat_inv) / K.dot(
                K.dot(tf.conj(d_theta), spat_cov_mat_inv), d_theta)
        result_fftd = w_theta * ffts
        sig_summed = K.sum(result_fftd, axis=1)
        frame += 1
        ### END OF PROGRAM
        self.input_feeds = [
            ffts,
            spat_cov_mat,
            frame,
            spec_avg,
            azimuth,
            elevation
        ]
        self.output_feeds = [
            sig_summed,
            spat_cov_mat,
            frame,
            spec_avg_bis,
            azimuth,
            elevation
        ]
        print("Tensorflow program built")

    def initialize(self):
        self.calc_all_angles()
        self.program()

    def process(self, ffts):
        with K.get_session() as sess:
            values = sess.run(
                self.input_feeds,
                feed_dict = zip(self.output_feeds, ...)
            )

if __name__ == "__main__":
    mics_locs = [[0.00000001, 0.00000001, 0.00000001],
                [0.1, 0.00000001, 0.00000001],
                [0.2, 0.00000001, 0.00000001],
                [0.00000001, -0.19, 0.00000001],
                [0.1, -0.19, 0.00000001],
                [0.2, -0.19, 0.00000001]]
    model = Model(6, 16000, 340, mics_locs, 128, 1024, 0.95)
    model.initialize()
