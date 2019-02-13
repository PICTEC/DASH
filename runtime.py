#!/usr/bin/env python

import argparse
import logging
import time
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dash.runtime")

from audio import Audio
from model import DolphinModel, NullModel
from mvdr_model import Model as MVDRModel
from mono_model import MonoModel
from post_filter import DAEPostFilter, NullPostFilter
from utils import fft, Remix

TIMEIT = None

MODEL_LIB = {
    "beam": MVDRModel,
    "dolphin": DolphinModel,
    "null": NullModel,
    "mono": MonoModel
}

POST_FILTER_LIB = {
    "dae": DAEPostFilter,
    "null": NullPostFilter
}

def main(audio_config, post_filter_config, model_config):
    """
    Main processing loop, all processing should be there, all configuration
    should be elsewhere, training should be done in other files.
    """
    audio = Audio(**audio_config)
    model_mode = model_config.pop("mode")
    model = MODEL_LIB[model_mode](**model_config)
    pf_mode = post_filter_config.pop("mode")
    post_filter = POST_FILTER_LIB[pf_mode](**post_filter_config)
    remixer = Remix(buffer_size=audio.buffer_size, buffer_hop=audio.buffer_hop,
                    channels=audio.n_out_channels)
    with audio:
        audio.open()
        model.initialize()
        post_filter.initialize()
        while True:
            if TIMEIT:
                ft = time.time()
                t = time.time()
            sample = audio.get_input()
            sample = fft(sample, audio.buffer_size, audio.n_in_channels)
            if TIMEIT:
                logger.info("Acquisition and FFT times {}ms".format(1000 * (time.time() - t)))
                t = time.time()
            sample = model.process(sample)
            if TIMEIT:
                logger.info("Model time {}ms".format(1000 * (time.time() - t)))
                t = time.time()
            sample = post_filter.process(sample)
            if TIMEIT:
                logger.info("Postfiltering time {}ms".format(1000 * (time.time() - t)))
                t = time.time()
            sample = remixer.process(sample)
            audio.write_to_output(sample)
            if TIMEIT:
                logger.info("Resampling and output {}ms".format(1000 * (time.time() - t)))
                logger.info("Iteration runtime {}ms".format(1000 * (time.time() - ft)))

def get_args():
    parser = argparse.ArgumentParser(description="Showcase of speech enhancement technologies.\n\n"
                                                 "To use, supply at least one of the configs from"
                                                 "`./configs`. In case you\n"
                                                 "want to use defaults, use -defaults flag.")
    parser.add_argument('-audio_config', help='path to audio config yaml file')
    parser.add_argument('-post_filter_config', help='path to post filter config yaml file')
    parser.add_argument('-model_config', help='path to model config yaml file')
    parser.add_argument('-defaults', help='Use if no other flag is used', action='store_true')
    parser.add_argument('-timeit', help='Indicate whether to time everything in the loop', action='store_true')
    args = parser.parse_args()
    if all([not x for x in [args.audio_config, args.post_filter_config, args.model_config, args.defaults]]):
        parser.print_help()
        exit(1)
    return args


if __name__ == "__main__":
    """
    If we need something, we may read those parameters from cmdline and pass it
    to the constructors of appropriate parts of the model. Most probably used for
    passing model.h5 filenames.
    """
    args = get_args()
    try:
        with open(args.audio_config, 'r') as file:
            audio_config = yaml.load(file)
    except:
        audio_config = {}
    try:
        with open(args.post_filter_config, 'r') as file:
            post_filter_config = yaml.load(file)
    except:
        post_filter_config = {"mode": "null", "fname": "storage/model-dae.h5"}
    try:
        with open(args.model_config, 'r') as file:
            model_config = yaml.load(file)
    except:
        model_config = {"mode": "beam", "n": 6, "f": 16000, "speed_of_sound": 340,
            "frame_hop": 128, "frame_len": 1024, "mu_cov": 0.95,
            "mics_locs": [[0.00000001, 0.00000001, 0.00000001],
                          [0.1, 0.00000001, 0.00000001],
                          [0.2, 0.00000001, 0.00000001],
                          [0.00000001, -0.19, 0.00000001],
                          [0.1, -0.19, 0.00000001],
                          [0.2, -0.19, 0.00000001]]}
    TIMEIT = args.timeit
    main(audio_config, post_filter_config, model_config)
