import numpy as np
import timeit

###
num_of_mics = 8
fft_len = 257
###
eigvec = np.ones((fft_len, num_of_mics), dtype=np.float32)/num_of_mics

def ev_by_power_iteration(fft_vector, eigenvector, speechMask):
    # R code
    # Tracking steering vector for this freq
    # if (speechActive[i_time, i_freq]) {
    # PSD_speech < - PSDTrackingConstant_speech * PSD_speech +
    # (1 - PSDTrackingConstant_speech) * fftVector % * % H(fftVector)
    # }
    # unnormalizedEigvec < - PSD_speech % * % eigenvector
    # eigenvector < - unnormalizedEigvec / sqrt(sum(abs(unnormalizedEigvec)) ^ 2 / n_channels)

    # which PSDs will be updated
    toUpd = speechMask == 1
    PSD_speech[toUpd] = PSDTrackingConstant_speech * PSD_speech[toUpd] + (1 - PSDTrackingConstant_speech) *\
                        np.dot(fft_vector[toUpd, :], fft_vector[toUpd, :].conj().T)

    unnormalizedEigvec = np.einsum('...ij,...j->...i', PSD_speech, eigenvector)
    eigenvector = unnormalizedEigvec / np.sqrt(np.sum(np.abs(unnormalizedEigvec)) ** 2 / n_channels)

    return eigenvector

# Xv = np.zeros((fft_len, num_of_mics), dtype=np.float32)
# Xv = ev + np.einsum('...ij,...j->...i', fft_vector, ev)
# Xv = X.dot(ev)
# Xv = np.dot(ev, X)
# np.einsum('ij,ij->i', X, ev) # use this!!
# timeit.timeit("for f in range(fft_len):    Xv[f, :] = np.dot(X[f, :, :], ev[f, :])", "import numpy as np; num_of_mics = 8; fft_len = 257; X = np.random.rand(fft_len, 8, 8); ev = np.ones((fft_len, num_of_mics), dtype=np.float32)/num_of_mics; Xv = np.zeros((fft_len, num_of_mics), dtype=np.float32)", number=10000)
# timeit.timeit("Xv = np.einsum('...ij,...j->...i', X, ev)", "import numpy as np; num_of_mics = 8; fft_len = 257; X = np.random.rand(fft_len, 8, 8); ev = np.ones((fft_len, num_of_mics), dtype=np.float32)/num_of_mics; Xv = np.zeros((fft_len, num_of_mics), dtype=np.float32)", number=10000)