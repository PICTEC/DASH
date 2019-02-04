from keras.models import load_model
import numpy as np

from utils import BufferMixin, StopOnConvergence, save_model
import dae


def default_model(n_fft):
    """
    This model is a bit too large for Tegra, but it is proven
    """
    assert n_fft == 257, "Default model cannot handle non-257 fft sizes"
    input_lower = Input((None, 257), name="input_lf")
    layer = Lambda(K.expand_dims)(input_lower)
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(9, 1), padding='same', activation='linear')(layer))
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(1, 5), padding='same', activation='linear')(layer))
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
    layer = LeakyReLU(0.01)(Conv2D(12, kernel_size=(17, 1), padding=None, activation='linear')(layer))
    layer = TimeDistributed(Flatten())(layer)
    layer = LeakyReLU(0.01)(Dense(2 * n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01)(Dense(n_fft, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = LeakyReLU(0.01, name='hidden')(Dense(3 * n_fft // 4, kernel_regularizer=L1L2(l1=1e-5))(layer))
    layer = Dense(n_fft)(layer)
    mdl = Model(input_lower, layer)
    mdl.summary()
    return mdl


class DAEPostFilter(BufferMixin([17, 257], np.complex64)):
    """
    It is assumed that PostFilter preserves phase of original signal.
    """

    _all_imports = {}
    _all_imports.update(dae.imports)
    _models = {"default": default_model,
               "fast": fast_model}

    def __init__(self, fname="storage/dae-pf.h5"):
        self.model = load_model(fname, self._all_imports)
        # TODO: perform checks whether fft_bin_size is proper

    def initialize(self):
        pass

    # TODO: this given an ENORMOUS shift in buffers - to reimplement
    def process(self, sample):
        self.buffer.push(sample)
        predictive = np.log(np.abs(self.buffer.reshape([1, 17, 257])) ** 2)
        result = self.model.predict(predictive)
        result = result[0, 8, :]  # extract channel of interest
        result = result * np.exp(1j * np.angle(sample))  # restore phase information
        return result

    @classmethod
    def train(cls, model_config, train_X, train_Y, valid_ratio=0.1, path_to_save="storage/dae-pf.h5", n_fft=513):
        """
        This should create a model from some training script...
        train_X should be padded by 16 from the beginning of the recording...
        n_fft - determines the size of the network
        """
        spec = cls._models[model_config] if isinstance(model_config, str) else model_config
        model = spec()
        # prepare validation and training data
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

if __name__ == "__main__":
    [train_X, train_Y], [test_X, test_Y] = ...
    model_config = "fast"  # to be interchangeable
    model = DAEPostFilter.train(model_config, train_X, train_Y)
    DAEPostFilter.test(model, test_X, test_Y)
