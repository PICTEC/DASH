import keras
import keras.backend as K
import logging
import numpy as np
import tensorflow as tf
import time

logger = logging.getLogger("dash.mono_model")

class MonoModel:
    def __init__(self, path):
        logger.info("Loading TF model")
        self.model = keras.models.load_model("storage/sr-lstm-repaired.h5")
        logger.info("TF model loaded")

    def initialize(self):
        self.model.reset_states()
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()

    def process(self, sample):
        prep = sample[:, 0].reshape(1, 1, -1)
        prep = np.abs(prep)
        # response = self.model.predict_function._callable_fn([prep])[0]
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        # response -= response.min()
        # response /= response.max()
        response = response >= 0.5
        response = response[0, 0, :] * sample[:, 0]
        return response

