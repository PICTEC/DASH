import argparse
import keras.backend as K
from keras.callbacks import Callback
from keras.layers import Input, Lambda, LeakyReLU, Conv2D, TimeDistributed, \
    Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
import librosa
import numpy as np
import random

from utils import list_sounds, open_sound


def mk_model():
    input_lower = Input((None, 257), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(16, kernel_size=(9, 1), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(20, kernel_size=(1, 5), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(24, kernel_size=(9, 1), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(28, kernel_size=(1, 5), padding='same', activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(1024)(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(512)(layer))
    layer = LeakyReLU(0.01)(Dense(350)(layer))
    layer = Dense(257)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl


class StopOnConvergence(Callback):
    def __init__(self, max_repetitions=10):
        super().__init__()
        self.max_repetitions = max_repetitions

    def on_train_begin(self, logs=None):
        self.repetitions = 0
        self.last_loss = np.inf

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if loss > self.last_loss:
                self.repetitions += 1
            else:
                self.last_loss = loss
                self.repetitions = 0
            if self.repetitions > self.max_repetitions:
                self.model.stop_training = True


class Simulator:
    def __init__(self):
        self.n_records = 3200
        self.valid = 160
        self.test = 160
        self.train = self.n_records - self.valid - self.test

    def load(self, path):
        fnames = list_sounds(path)
        fnames = random.sample(fnames, self.n_records)
        max_len = max([len(open_sound(x)[1]) for x in fnames])
        max_len = 1 + (max_len - 512) // 128
        self.X = np.ones([self.n_records, max_len, 257], np.float32)
        self.Y = np.ones([self.n_records, max_len, 257], np.float32)
        self.X *= np.log(2e-12)
        self.Y *= np.log(2e-12)
        for ix, fname in enumerate(fnames):
            sr, rec = open_sound(fname)
            assert sr == 16000
            rec = np.log(2e-12 + np.abs(librosa.stft(rec.astype(np.float32) / (2**15), n_fft=512, hop_length=128).T[:max_len]) ** 2)
            self.X[ix, :len(rec)] = self.mask(rec)
            self.Y[ix, :len(rec)] = rec
        return ([self.X[:self.train], self.Y[:self.train]],
                [self.X[self.train:self.train+self.valid], self.Y[self.train:self.train+self.valid]],
                [self.X[-self.test:], self.Y[-self.test:]])

    def mask(self, spec):
        random_values = np.random.random(spec.shape)
        mask = -50 * np.random.random(spec.shape)
        mask = mask > spec
        return mask * random_values + (1 - mask) * spec


def save_model(model, path):
    """
    Model should be stripped of all callbacks and needless objects...
    """
    model.save(path)


def training(dataset, path):
    [train_X, train_Y], [valid_X, valid_Y], [test_X, test_Y] = dataset
    model = mk_model()
    # backup_callback = BackupCallback()
    for lr in [0.001, 0.0001, 0.00001]:
        model.compile(optimizer=Adam(lr, clipnorm=1.), loss='mse')
        model.fit(train_X, train_Y, validation_data=[valid_X, valid_Y], epochs=100,
                    callbacks=[StopOnConvergence(3)], batch_size=8)  # , backup_callback])
    # test...
    save_model(model, path)  # create postfilter object...


imports = {"StopOnConvergence": StopOnConvergence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trainer of DAE")
    parser.add_argument("-t", nargs=1, help='Synthetize input data using this model (otherwise use built-in simulator)')
    parser.add_argument("-p", nargs=1, help='Path from which to load recordings')
    parser.add_argument("-o", nargs=1, help='Where to store output')
    args = parser.parse_args()
    assert args.o is not None
    print(args)
    if args.t:
        loader = Loader(args.t[0])
    else:
        loader = Simulator()
    dataset = loader.load(args.p[0])
    training(dataset, path=args.o[0])
