#!/usr/bin/env python3

import argparse
import keras.backend as K
from keras.layers import Lambda, LeakyReLU, Conv2D, Flatten, TimeDistributed, Dense, Input
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.regularizers import L1L2
import numpy as np
import os
import time

from utils import BufferMixin, StopOnConvergence, save_model, list_sounds, open_sound, stft
import dae


def default_model(n_fft):
    """
    This model is a bit too large for Tegra, but it is proven
    """
    assert n_fft == 257, "Default model cannot handle non-257 fft sizes"
    input_lower = Input((None, 257), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(1024, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(512, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(350, kernel_regularizer=L1L2(l2=1e-5))(layer))
    layer = Dense(257)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl


def fast_model(n_fft):
    """
    Architecture of a faster model
    """
    input_lower = Input((None, n_fft), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(17, 1), activation='linear')(layer))
    layer = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(layer)
    layer = LeakyReLU(0.01)(Dense(n_fft // 2)(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft // 4)(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(2 * n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(3 * n_fft // 4, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Dense(n_fft)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl

def faster_model(n_fft):
    """
    Architecture of a faster model
    """
    input_lower = Input((None, n_fft), name="input_lf")
    layer = LeakyReLU(0.01)(Dense(n_fft, kernel_regularizer=L1L2(l1=1e-5))(input_lower))
    layer = LeakyReLU(0.01)(Dense(2 * n_fft // 3, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft // 2, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Lambda(lambda x: K.expand_dims(x))(layer)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(17, 1), activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01, name='hidden')(Dense(3 * n_fft // 4, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Dense(n_fft)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl



class DAEPostFilter(BufferMixin([17, 257, 1], np.complex64)):
    """
    It is assumed that PostFilter preserves phase of original signal.
    """

    _all_imports = {}
    _all_imports.update(dae.imports)
    _models = {"default": default_model,
               "fast": fast_model,
               "faster": faster_model}

    def __init__(self, fname="storage/dae-pf.h5", n_fft=1024):
        super().__init__()
        self.model = load_model(fname, self._all_imports)
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()
        fft_size = n_fft // 2 + 1
        assert self.model.input_shape[-1] == fft_size, "Input shape is {}; model requires {}".format(fft_size, self.model.input_shape[-1])
        assert self.model.output_shape[-1] == fft_size, "Input shape is {}; model requires {}".format(fft_size, self.model.output_shape[-1])

    def initialize(self):
        pass

    def process(self, sample):
        self.buffer.push(sample)
        predictive = -np.log(np.abs(self.buffer.reshape([1, 17, 257])) ** 2 + 2e-12)
        result = self.session.run(self.output,
                    feed_dict={self.input: predictive})
        result = result[:1, 0, :].T  # extract channel of interest
        result = np.sqrt(np.exp(-result)) * np.exp(1j * np.angle(sample))  # restore phase information
        return result

    @classmethod
    def train(cls, model_config, train_X, train_Y, valid_ratio=0.1, path_to_save="storage/dae-pf.h5", n_fft=512):
        """
        This should create a model from some training script...
        train_X should be padded by 16 from the beginning of the recording...
        n_fft - determines the size of the network
        """
        fft_size = n_fft // 2 + 1
        spec = cls._models[model_config] if isinstance(model_config, str) else model_config
        model = spec(fft_size)
        sel = np.random.random(len(train_X)) > valid_ratio
        train_X, valid_X = train_X[sel], train_X[~sel]
        train_Y, valid_Y = train_Y[sel], train_Y[~sel]
        for lr in [0.0003, 0.0001, 0.00003]:
            model.compile(optimizer=Adam(lr, clipnorm=1.), loss='mse')
            model.fit(train_X, train_Y, validation_data=[valid_X, valid_Y], epochs=50,
                        callbacks=[StopOnConvergence(5)], batch_size=8)
        save_model(model, path_to_save)
        return model

    @staticmethod
    def test(model, test_X, test_Y):
        pass


class NullPostFilter:
    def initialize(self):
        pass

    def process(self, sample):
        return sample


def list_dataset(clean, noisy):
    cleans = set([x.split(os.sep)[-1] for x in list_sounds(clean)])
    noises = set([x.split(os.sep)[-1] for x in list_sounds(noisy)])
    fnames = cleans | noises
    return [os.path.join(clean, x) for x in fnames], [os.path.join(noisy, x) for x in fnames]

def get_dataset(clean, noisy, ratio=0.2, maxlen=1200, n_fft=512):
    fft_size = n_fft // 2 + 1
    clean, noisy = list_dataset(clean, noisy)
    assert clean, "No data with common filenames"
    assert noisy, "No data with common filenames"
    X = np.zeros([len(clean), maxlen + 16, fft_size], np.float32)
    Y = np.zeros([len(clean), maxlen, fft_size], np.float32)
    sel = np.random.random(len(clean)) > ratio
    for ix, (cl, ns) in enumerate(zip(clean, noisy)):
        print("Loading file", ix)
        cl, ns = open_sound(cl), open_sound(ns)
        assert cl[0] == ns[0]
        cl, ns = cl[1], ns[1]
        if len(ns.shape) > 1:
            ns = ns[:, 0]
        spec = -np.log(np.abs(stft(cl, n_fft=n_fft)) ** 2 + 2e-12).T[:maxlen]
        spec = np.pad(spec, ((16, maxlen - spec.shape[0]), (0, 0)), 'constant', constant_values=-np.log(2e-12))
        X[ix, :, :] = spec
        spec = -np.log(np.abs(stft(ns, n_fft=n_fft)) ** 2 + 2e-12).T[:maxlen]
        spec = np.pad(spec, ((0, maxlen - spec.shape[0]), (0, 0)), 'constant', constant_values=-np.log(2e-12))
        Y[ix, :, :] = spec
    return [X[sel], Y[sel]], [X[~sel], Y[~sel]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply both path parameters")
    parser.add_argument("--noisy", help="Path to noisy (source) recordings")
    parser.add_argument("--clean", help="Path to clean recordings")
    args = parser.parse_args()
    if args.clean is None or args.noisy is None:
        parser.print_help()
        exit(1)
    [train_X, train_Y], [test_X, test_Y] = get_dataset(args.clean, args.noisy)
    model_config = "default"  # to be interchangeable
    model = DAEPostFilter.train(model_config, train_X, train_Y)
    DAEPostFilter.test(model, test_X, test_Y)
