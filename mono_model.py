import keras
import logging
import numpy as np
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

    def process(self, sample):
        prep = sample[:, 0].reshape(1, 1, -1)
        prep = np.abs(prep)
        response = self.model.predict_function([prep])[0]
        # response -= response.min()
        # response /= response.max()
        response = response >= 0.5
        response = response[0, 0, :] * sample[:, 0]
        return response

