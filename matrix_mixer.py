#!/usr/bin/python3

# Usage:
# python matrix_mixer.py --pos '[[0, 0.1, 0.0],[0,0,0],[0,-0.1,0],[0.19,0.1,0],[0.19,0,0],[0.19, -0.1, 0.0]]' --src DAE-libri-cp --nexamples 3000 --nsrc 1 --ndiff 1 --diff bin/backgrounds --bgamp 1.2 --srcpos '[[0, 1.5, 1.5]]' --srclist 1

import argparse
import json
import librosa
import numpy as np
import os
import random
import scipy.io.wavfile as sio

from utils import list_sounds, open_sound


class Source:
    def __init__(self, source, sr=16000):
        if isinstance(source, str):
            self.sr, self.signal = open_sound(source)
        elif isinstance(source, np.ndarray):
            self.signal = source
        else:
            raise TypeError("Unsupported argument type {}".format(type(source)))
        assert len(self.signal.shape) == 1

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, v):
        self.__setattr__(key, v)

    @property
    def duration(self):
        return len(self.signal)


class Diffuse:
    def __init__(self, source, sr=16000, amp=1.0):
        if isinstance(source, str):
            self.sr, self.signal = open_sound(source)
            self.signal = self.signal.T
        elif isinstance(source, np.ndarray):
            self.signal = source
            self.sr = sr
        else:
            raise TypeError("Unsupported argument type {}".format(type(source)))
        self.signal = self.signal * amp
        self.channels = self.signal.shape[0]  # TODO: correct index?
        print(self.channels)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, v):
        self.__setattr__(key, v)

    @property
    def length(self):
        return self.signal.shape[1]


class Matrix:
    def __init__(self, positions):
        self.positions = np.array(positions)
        self.n_channels = len(self.positions)

class Scene:
    def __init__(self, sound_speed = 340, sample_rate=16000):
        self.sound_speed = 340
        self.sample_rate = 16000
        self.sources = []
        self.diffuse = []

    def add_diffuse(self, diffuse, random_start=True):
        diffuse["random_start"] = random_start
        self.diffuse.append(diffuse)

    def add_source(self, source, position, start=0):
        source.position = np.array(position)
        source.start = 0
        self.sources.append(source)

    def render(self, matrix, pad_diffuse=False, align_delays=True, gain_scaling=False):
        # TODO: resample all signals and enforce uniform coding (e.g. np.float32)
        # determine lengths
        ends = [src["start"] + src["duration"] for src in self.sources]
        max_end = max(ends)
        # if diffuse is not long enough, do something with it
        if any([src["length"] < max_end for src in self.diffuse]):
            if pad_diffuse:
                for src in self.diffuse:
                    if src["length"] < max_end:
                        src["signal"] = np.pad(
                            src["signal"], ((0, 0), (0, max_end - src["length"])), 'constant'
                        )
                        src["length"] = max_end
            else:
                raise TypeError("Diffuse sources are too short!")
        # create zeroed channels
        sound = np.zeros([matrix.n_channels, max_end], np.float32)
        # crop diffuse to proper lengths and mix them
        for src in self.diffuse:
            if src["random_start"]:
                begin = np.random.randint(src["length"] - max_end + 1)
            else:
                begin = 0
            end = begin + max_end
            sound += src["signal"][:, begin:end]
        # calculate transmission gains and delays
        gains = np.zeros((len(matrix.positions), len(self.sources)))
        delays = np.zeros((len(matrix.positions), len(self.sources)))
        for mic_ix, pos in enumerate(matrix.positions):
            for src_ix, source in enumerate(self.sources):
                print(source.position)
                distance = np.sqrt(np.sum((source.position - pos)**2))
                if gain_scaling:
                    dist_ratio = distance / source.reference_distance
                    gains[mic_ix, src_ix] = dist_ratio ** -2
                else:
                    gains[mic_ix, src_ix] = 1
                print("Distance:", distance)
                print("TDOA:", distance / self.sound_speed * self.sample_rate)
                delays[mic_ix, src_ix] = distance / self.sound_speed * self.sample_rate
        if align_delays:
            delays -= delays.min()
        print(delays)
        for mic_ix in range(len(matrix.positions)):
            for src_ix, source in enumerate(self.sources):
                signal = gains[mic_ix, src_ix] * source.signal
                signal = self.fractional_delay(signal, delays[mic_ix, src_ix])
                sound[mic_ix, source.start:source.start+source.duration] += signal
        return sound

    def resample(self, signal, source_sample_rate):
        return librosa.resample(signal, source_sample_rate, self.sample_rate)

    def fractional_delay(self, signal, delay, level=30):
        # uses non-causal ideal fractional delay filter
        filter = np.sinc(np.arange(2 * level + 1) - level - delay)
        return np.convolve(signal, filter, mode='same')


def save_sound(sound, fname, rate):
    """
    Saves in 16kHz, 16bit, performing necessary limiting
    """
    sound = sound * 2**15
    sound = sound.astype(np.int16)
    print(sound)
    print(rate)
    print(fname)
    sio.write(fname, rate, sound.T)

def iterate_src_pos(sources, positions, n):
    return zip(random.sample(sources, n), random.sample(positions, n))

def create_dataset(sources, positions, diffuse, n_src, n_examples, target_fname,
                   n_diff, matrix, bgamp):
    for i in range(n_examples):
        scene = Scene()
        for src, pos in iterate_src_pos(sources, positions, n_src):
            speaker = Source(src)
            scene.add_source(speaker, pos)
        for diff in random.sample(diffuse, n_diff):
            diff = Diffuse(diff, amp=bgamp)
            scene.add_diffuse(diff)
        sound = scene.render(matrix)
        save_sound(sound, target_fname.format(i), scene.sample_rate)

def by_list(sources, positions, diffuse, n_src, n_examples, target_fname,
                   n_diff, matrix, bgamp):
    spec = []
    spec.append("Background amplification: {}, sources per recording: {}".format(
        bgamp, n_src))
    fnames = [target_fname.format(i) for i in range(n_examples)]
    fnames = list(sorted(fnames))
    for fname in fnames:
        scene = Scene()
        batch, sources = sources[:n_src], sources[n_src:]
        batch_pos = random.sample(positions, n_src)
        batch_diff = random.sample(diffuse, n_diff)
        for src, pos in zip(batch, batch_pos):
            speaker = Source(src)
            scene.add_source(speaker, pos)
        for diff in batch_diff:
            diff = Diffuse(diff, amp=bgamp)
            scene.add_diffuse(diff)
        sound = scene.render(matrix)
        save_sound(sound, fname, scene.sample_rate)
        spec.append("{} : {} with diffuse {} ".format(fname,
            ["{} @ {}".format(x, y) for x, y in zip(batch, batch_pos)],
            batch_diff))
    with open("spec.txt", "w") as f:
        f.write("\n".join(spec))


def main():
    parser = argparse.ArgumentParser(
        description="Creates a dataset of artificial multi-channel mixtures",
        epilog="Matrix positions and source files have to be specified for the script to run")
    parser.add_argument("--pos", nargs=1, help="Specify matrix microphone positions, in metres")
    parser.add_argument("--srcpos", nargs='+', help="Source position -- each variable parses"
                                                    "to one position; if more than sources, the"
                                                    "position choices are randomized")
    parser.add_argument("--ndiff", nargs=1, help="Specify how much diffuse sources to use")
    parser.add_argument("--diff", nargs='+', help="Paths for diffuse files (may be directories)")
    parser.add_argument("--nsrc", nargs=1, help="Specify how much directed sources to use")
    parser.add_argument("--src", nargs='+', help="Paths for source files (may be directories)")
    parser.add_argument("--gain", help="Whether to calculate gains of the sources")
    parser.add_argument("--srclist", help="Instead of randomizing sources, use all of them")
    parser.add_argument("--nexamples", nargs=1, help="Number of examples in the dataset")
    parser.add_argument("--bgamp", nargs=1, help="Amplify diffuse sources by this amount")
    args = parser.parse_args()
    if args.pos is None:
        print(parser.format_usage())
        print("FATAL: position of matrix not specified")
        exit(1)
    matrix = Matrix(json.loads(args.pos[0]))
    sources = []
    if args.src is None:
        print(parser.format_usage())
        print("FATAL: source files not specified")
        exit(1)
    for src in args.src:
        if src.endswith(".wav") or src.endswith(".flac"):
            sources.append(src)
        else:
            sources += list_sounds(src)
    sources = list(sorted(sources))
    print(sources)
    diffuse = []
    if args.diff is None:
        if args.ndiff is not None and int(args.ndiff[0]) > 0:
            print(parser.format_usage())
            print("FATAL: specify diffuse sources if any are used")
            exit(1)
    else:
        for src in args.diff:
            if src.endswith(".wav") or src.endswith(".flac"):
                diffuse.append(src)
            else:
                diffuse += list_sounds(src)
    print(diffuse)
    positions = []
    if args.srcpos is not None:
        for src in args.srcpos:
            positions = list(np.array(json.loads(src)))
    else:
        positions = [np.random.random(3) - 0.5 for x in range(10)]
        positions = [x / np.sum(x) for x in positions]
    print("Sources:", len(sources))
    print(positions)
    ndiff = int(args.ndiff[0]) if args.ndiff else 0
    nsrc = int(args.nsrc[0]) if args.nsrc else 1
    nexamples = int(args.nexamples[0]) if args.nexamples else 10
    bgamp = float(args.bgamp[0]) if args.bgamp else 1
    try: os.mkdir("dataset")
    except: pass
    if args.srclist:
        by_list(sources, positions, diffuse, nsrc, n_examples=nexamples,
            target_fname="dataset/{}.wav", n_diff=ndiff, matrix=matrix, bgamp=bgamp)
    else:
        create_dataset(sources, positions, diffuse, nsrc, n_examples=nexamples,
            target_fname="dataset/{}.wav", n_diff=ndiff, matrix=matrix, bgamp=bgamp)




if __name__ == "__main__":
    main()
