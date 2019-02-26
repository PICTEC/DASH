import keras
import keras.backend as K
import logging
import numpy as np
import tensorflow as tf
import time

logger = logging.getLogger("dash.mono_model")


class MonoModel:
    """
    MonoModel wraps monophonic masking into a simple model usage within Runtime.
    This models loads its' neural network from `path` and prepares it for fast
    evaluation. Model is assumed to accept plain absolute values of spectrum and
    return a soft mask for that spectrum. Masks are scaled and clipped before
    application to the data.
    """
    
    def __init__(self, path, scaling_factor=1, clip=0):
        logger.info("Loading TF model")
        self.clip = clip
        self.model = keras.models.load_model(path)
        logger.info("TF model loaded")
        self.scaling_factor = float(scaling_factor)

    def initialize(self):
        """
        Prepare the model - load all required data.
        """
        self.model.reset_states()
        self.model._make_predict_function()
        self.input = self.model.input
        self.output = self.model.output
        self.session = K.get_session()

    def process(self, sample):
        """
        Accept a single multichannel frame. Discard all but one channel
        and perform masking on that channel.
        """
        prep = sample[:, 0].reshape(1, 1, -1)
        prep = np.abs(prep)
        response = self.session.run(self.output,
            feed_dict={self.input: prep})
        response = np.clip(response, 0, None) ** self.scaling_factor
        response[response < self.clip] = 0
        response = response[0, 0, :] * sample[:, 0]
        return response.reshape(-1, 1)



class MonoModelWindowed:
    pass