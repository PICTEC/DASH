#!/usr/bin/env python3
import threading
from queue import Queue
import pyaudio
import numpy as np

class play_thread(threading.Thread):
    """

    Args:
        buffer (queue.Queue):
        hop (int):
        sample_rate (int):
        channels (int):
    """
    def __init__(self, buffer, hop, sample_rate, channels):
        super(play_thread, self).__init__()

        self.buffer = buffer
        self.hop = hop
        self.sample_rate = sample_rate
        self.channels = channels

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  output=True)

    def run(self):
        """
        """
        while not self._is_stopped:
            vals = np.array([self.buffer.get() for _ in range(self.hop)])
            self.stream.write(frames=vals, num_frames=self.hop)

class Audio:
    """

    Application can create an instance of this class and pass it to models.
    The application can use it's methods (or methods as callbacks) to fetch
    some more data for next iteration of a model.
    Class should utilize sliding buffer, so user needs only to read that buffer, STFT it
    and the processing is set up.

    Args:
        buffer_size (int, optional): number of samples in a single output frame
        buffer_hop (int, optional): number of samples that gets removed from a buffer on a single read
        sample_rate (int, optional): self explanatory
        n_in_channels (int, optional): number of input channels
        n_out_channels (int, optional): number of output channels
    """

    def __init__(self, buffer_size=512, buffer_hop=128, sample_rate=16000, n_in_channels=6,
                 n_out_channels=1):
        self.buffer_size = buffer_size
        self.buffer_hop = buffer_hop
        self.sample_rate = sample_rate
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels

        self.in_buffers = [Queue(maxsize=buffer_size)] * n_in_channels
        self.out_buffers = [Queue(maxsize=buffer_size)] * n_out_channels

    def open(self):
        """

        Opens an audio device and sets up a thread that reads the data from audio device
        to a buffer, and sets up a buffer for playing the sound for the user.
        """

    def close(self):
        """Closes all audio devices, playing what's left in the play buffer
        """

    def listen(self):
        """

        Reads the top buffer_size samples from recording buffer, removes top buffer_hop samples
        from the read buffer and returns the recorded sound frame to the user

        Returns:
            [buffer_size, n_in_channels] of type np.float32 (32-bit wav in [-1, 1] range)
        """

    def play(self, output):
        """Appends the data to the play buffer, from which the sound driver plays the sound
        Accepts: [*any int*, n_out_channels] of type np.float32 (32-bit wav in [-1, 1] range)

        Args:
            output ():
        """


    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, tb):
        # TODO: handle exception
        return self.close()


if __name__ == "__main__":
    """
    Simple test of whether the class works - a single-channel loopback recording
    """
    audio = Audio(n_in_channels=1, n_out_channels=1, buffer_size = 128)
    with audio:
        while True:
            audio.play(audio.listen())
