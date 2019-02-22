import keras
import keras.backend as K
import logging
import numpy as np
import tensorflow as tf
import time

logger = logging.getLogger("dash.mono_model")

class MonoModel:
    def __init__(self, path, scaling_factor):
        logger.info("Loading TF model")
        self.model = keras.models.load_model(path)
        logger.info("TF model loaded")
        self.scaling_factor = float(scaling_factor)

    def initialize(self):
        self.model.reset_states()
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()

    def process(self, sample):
        prep = sample[:, 0].reshape(1, 1, -1)
        prep = np.abs(prep)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        response = np.clip(response, 0, None)
        response = response[0, 0, :] * sample[:, 0]
        return response.reshape(-1, 1)
