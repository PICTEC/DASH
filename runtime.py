#!/usr/bin/env python

import argparse
import gc
import json
import logging
import numpy as np
import paho.mqtt.client as mqtt
import random
import time
import yaml
from os import listdir

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dash.runtime")

from audio import Audio
from model import DolphinModel, NullModel
from mvdr_model import Model as MVDRModel
from mono_model import MonoModel, MonoModelWindowed
from post_filter import DAEPostFilter, NullPostFilter
from utils import fft, Remix, AdaptiveGain
import cProfile
import re

MODEL_LIB = {
    "beam": MVDRModel,
    "dolphin": DolphinModel,
    "null": NullModel,
    "mono": MonoModel,
    "mono-pf": MonoModelWindowed
}

POST_FILTER_LIB = {
    "dae": DAEPostFilter,
    "null": NullPostFilter
}


class Runtime:
    def __init__(self):
        self.configurations = {
            "lstm-mvdr": {
                "name": "Deep MVDR",
                "configs": [
                    "configs/large_hop_input_config.yaml",
                    "configs/null_postfilter.yaml",
                    "configs/lstm_mvdr_config.yaml"
                ]
            },
            "postfilter": {
                "name": "Monophonic Denoising Autoencoder",
                "configs": [
                    "configs/large_hop_input_config.yaml",
                    "configs/postfilter.yaml",
                    "configs/null_model.yaml"
                ]
            },
            "small-lstm": {
                "name": "Monophonic LSTM Masking - v1",
                "configs": [
                    "configs/large_hop_input_config.yaml",
                    "configs/null_postfilter.yaml",
                    "configs/1_layer.yaml"
                ]
            },
            "larger-lstm": {
                "name": "Monophonic LSTM Masking - v2",
                "configs": [
                    "configs/large_hop_input_config.yaml",
                    "configs/null_postfilter.yaml",
                    "configs/3_layer.yaml"
                ]
            },
            "dae-lstm": {
                "name": "Monophonic LSTM Masking - v3",
                "configs": [
                    "configs/large_hop_input_config.yaml",
                    "configs/null_postfilter.yaml",
                    "configs/pf_masking_layer.yaml"
                ]
            },
        }
        self.default = "lstm-mvdr"
        self.TIMEIT = None
        self.audio = None
        self.play_processed = True
        self.enabled = True
        self.client = mqtt.Client("dash.runtime")
        self.client.user_data_set(self)
        def callback(client, self, message):
           self.Q.append(message)
        self.client.on_message = callback
        self.client.connect('localhost')
        self.client.subscribe('dash.control')
        self.client.loop_start()
        config = [self.default, {v["name"]: k for k,v in self.configurations.items()}]
        self.client.publish("dash.config", json.dumps(config), retain=True)
        self.Q = []

    def send_message(self, in_spec, out_spec, location):
        self.client.publish("dash.in", in_spec.tobytes())
        self.client.publish("dash.out", out_spec.tobytes())
        if location is not None:
            self.client.publish("dash.pos", location)
        else:
            self.client.publish("dash.pos", random.random())

    def check_queue(self):
        """
        If message to change model - call rebuild, if flag is changed - change proper variable

        TODO: start/stop messages
        """
        if self.Q:
            message = self.Q.pop().payload
            logger.info(message)
            if message.startswith(b"START"):
                self.enabled = True
            elif message.startswith(b"STOP"):
                self.pause()
            elif message.startswith(b"SWITCH"):
                self.rebuild(message[7:].decode("ascii"))
            elif message.startswith(b"PLAY"):
                if message[5:] == b"IN":
                    self.play_processed = False
                elif message[5:] == b"OUT":
                    self.play_processed = True

    def pause(self):
        self.enabled = False
        self.audio.close()
        while not self.enabled:
            self.check_queue()
            time.sleep(0)
        self.audio.open()

    def rebuild(self, config):
        audio_config, post_filter_config, model_config = [yaml.load(open(x)) for x in self.configurations[config]["configs"]]
        self.audio.close()  # this should clean all buffers
        self.build(audio_config, post_filter_config, model_config)
        self.audio.open()

    def build(self, audio_config, post_filter_config, model_config):
        if self.audio is None:
            self.audio = Audio(**audio_config)
        model_mode = model_config.pop("mode")
        self.model = MODEL_LIB[model_mode](**model_config)
        pf_mode = post_filter_config.pop("mode")
        self.post_filter = POST_FILTER_LIB[pf_mode](**post_filter_config)
        self.model.initialize()
        self.post_filter.initialize()
        # TODO: initialize FFT properly

    def main(self, audio_config=None, post_filter_config=None, model_config=None):
        """
        Main processing loop, all processing should be there, all configuration
        should be elsewhere, training should be done in other files.
        """
        if audio_config is None or post_filter_config is None or model_config is None:
            audio_config, post_filter_config, model_config = [yaml.load(open(x)) for x in self.configurations[self.default]["configs"]]
        self.build(audio_config, post_filter_config, model_config)
        in_gain = AdaptiveGain(level=0.005, max_gain=10)
        out_gain = AdaptiveGain()
        remixer = Remix(buffer_size=self.audio.buffer_size, buffer_hop=self.audio.buffer_hop,
                        channels=self.audio.n_out_channels)
        # initialize FFT
        fft(np.random.random([self.audio.buffer_size, self.audio.n_in_channels]), self.audio.buffer_size, self.audio.n_in_channels)
        with self.audio:
            while True:
                if self.TIMEIT:
                    ft = time.time()
                    t = time.time()
                sample = self.audio.get_input()
                sample = in_gain.process(sample)
                if self.TIMEIT:
                    logger.info("Acquisition time {}ms".format(1000 * (time.time() - t)))
                    t = time.time()
                in_sample = sample = fft(sample, self.audio.buffer_size, self.audio.n_in_channels)
                if self.TIMEIT:
                    logger.info("FFT time {}ms".format(1000 * (time.time() - t)))
                    t = time.time()
                sample = self.model.process(sample)
                if self.TIMEIT:
                    logger.info("Model time {}ms".format(1000 * (time.time() - t)))
                    t = time.time()
                out_plot = sample = self.post_filter.process(sample)
                if self.TIMEIT:
                    logger.info("Postfiltering time {}ms".format(1000 * (time.time() - t)))
                    t = time.time()
                if self.play_processed:
                    sample = remixer.process(sample[:, 0])
                    # sample = out_gain.process(sample)
                    self.audio.write_to_output(sample)
                else:
                    sample = remixer.process(in_sample[:, 0])
                    # sample = out_gain.process(sample)
                    self.audio.write_to_output(sample)
                if self.TIMEIT:
                    logger.info("Resampling and output {}ms".format(1000 * (time.time() - t)))
                    logger.info("Iteration runtime {}ms".format(1000 * (time.time() - ft)))
                self.check_queue()
                self.send_message(in_sample, out_plot, None)


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
    parser.add_argument('-input_from_catalog', help='Process all wav files in given directory')
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
        print("Cannot load audio config {}, exiting".format(args.audio_config))
        raise
        exit(1)
    try:
        with open(args.post_filter_config, 'r') as file:
            post_filter_config = yaml.load(file)
    except:
        print("Cannot load post filter config, exiting")
        exit(1)
    try:
        with open(args.model_config, 'r') as file:
            model_config = yaml.load(file)
    except:
        print("Cannot load model config, exiting")
        exit(1)

    if args.input_from_catalog:
        waves = [file for file in listdir(args.input_from_catalog) if ".wav" in file]
        for wave in waves:
            gc.collect()
            audio_config['input_from_file'] = args.input_from_catalog + '/' + wave
            audio_config['record_name'] = wave
            try:
                print('Processing: ', audio_config['input_from_file'])
                runtime = Runtime()
                runtime.TIMEIT = args.timeit
                runtime.main(audio_config.copy(), post_filter_config.copy(), model_config.copy())
            except Exception as e:
                print(e)
            time.sleep(1)
    else:
        runtime = Runtime()
        runtime.TIMEIT = args.timeit
        runtime.main(audio_config, post_filter_config, model_config)
