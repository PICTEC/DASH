#!/usr/bin/env python

import argparse
import yaml

from audio import Audio
from model import DolphinModel
from post_filter import PostFilter
from utils import fft, Remix

def main(audio_config, post_filter_config, model_config):
    """
    Main processing loop, all processing should be there, all configuration
    should be elsewhere, training should be done in other files.
    """
    audio = Audio(**audio_config)
    post_filter = PostFilter(**post_filter_config)
    model = DolphinModel(**model_config)
    remixer = Remix()
    with audio:
        audio.open()
        model.initialize()
        post_filter.initialize()
        while True:
            sample = audio.get_input()
            sample = fft(sample)
            sample = model.process(sample)
            sample = post_filter.process(sample)
            sample = remixer.process(sample)
            audio.write_to_output(sample[0,:])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-audio_config', help='path to audio config yaml file')
    parser.add_argument('-post_filter_config', help='path to post filter config yaml file')
    parser.add_argument('-model_config', help='path to model config yaml file')
    return parser.parse_args()

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
        post_filter_config = {"mode": "dae", "fname": "storage/model-dae.h5"}
    try:
        with open(args.model_config, 'r') as file:
            audio_config = yaml.load(file)
    except:
        model_config = {}

    main(audio_config, post_filter_config, model_config)
