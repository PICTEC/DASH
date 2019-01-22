#!/usr/bin/env python3

import numpy as np
import pyaudio
from queue import Queue
import threading
import time
import wave
from os import makedirs

class PlayThread(threading.Thread):
    """Thread which pass data stored in the buffer to the speakers

    Args:
        p (pyAudio): Python interface to PortAudio
        buffer (queue.Queue): Queue with byte string to be played
        hop (int): Number of samples to play in single read
        sample_rate (int): Sample rate [Hz] of playing
        channels (int): Number of channels to play
        id (int, optional): Index of output Device to use
        record_to_file (bool, optional): Save played output also to the file
            stored in 'records/outputs/', default set to False
    """
    def __init__(self, p, buffer, hop, sample_rate, channels, id=None, record_to_file=False):
        super(PlayThread, self).__init__()

        self.buffer = buffer
        self.hop = hop
        self.sample_rate = sample_rate
        self.channels = channels
        self.daemon = True
        self.stopped = False
        self.record_to_file = record_to_file

        self.stream = p.open(format=pyaudio.paFloat32,
                             channels=self.channels,
                             rate=self.sample_rate,
                             output=True,
                             frames_per_buffer=self.hop,
                             output_device_index=id)

        if record_to_file:
            try:
                makedirs('records/outputs')
            except:
                pass
            file_name = 'records/outputs/' + time.asctime() + '_out.wav'
            self.f = wave.open(file_name, 'w')
            self.f.setnchannels(channels)
            self.f.setsampwidth(4)
            self.f.setframerate(sample_rate)

    def run(self):
        """Method representing the thread’s activity

        Wait until buffer is full, than play frames from the buffer until
        thread is stopped.
        """
        while not self.buffer.full():
            pass

        while not self.stopped:
            if not self.buffer.empty():
                frames = self.buffer.get()
                self.stream.write(frames=frames)
                if self.record_to_file:
                    self.f.writeframesraw(frames)

    def stop(self):
        """Stop thread, play what's left in the buffer and close stream
        """
        self.stopped = True

        while not self.buffer.empty():
            self.stream.write(frames=self.buffer.get())

        self.stream.close()
        if self.record_to_file:
            self.f.writeframes(b'')
            self.f.close()


class ReadThread(threading.Thread):
    """Thread which read data from microphones and pass it to the buffer

    Args:
        p (pyAudio): Python interface to PortAudio
        buffer (queue.Queue): Queue where write byte strings
        hop (int): Number of samples to record in single read
        sample_rate (int): Sample rate [Hz] of recording
        channels (int): Number of channels to record
        id (int, optional): Index of input Device to use
        from_file (str, optional): Path to the file from which read input, if not
            provided, than it will be get input from input audio device
        record_to_file (bool, optional): Save played output also to the file
            stored in 'records/outputs/', default set to False
    """
    def __init__(self, p, buffer, hop, sample_rate, channels, id=None,
                 from_file=None, record_to_file=True):
        super(ReadThread, self).__init__()

        self.buffer = buffer
        self.hop = hop
        self.sample_rate = sample_rate
        self.channels = channels
        self.daemon = True
        self.stopped = False
        self.record_to_file = record_to_file
        self.from_file = from_file

        if from_file is None:
            self.stream = p.open(format=pyaudio.paFloat32,
                                 channels=self.channels,
                                 rate=self.sample_rate,
                                 input=True,
                                 frames_per_buffer=self.hop,
                                 input_device_index=id)
        else:
            self.wf = wave.open(from_file, 'rb')

        if record_to_file:
            try:
                makedirs('records/inputs')
            except:
                pass
            file_name = 'records/inputs/' + time.asctime() + '_in.wav'
            self.f = wave.open(file_name, 'w')
            self.f.setnchannels(channels)
            self.f.setsampwidth(4)
            self.f.setframerate(sample_rate)

    def run(self):
        """Method representing the thread’s activity

        Get data from microphones or from file and put it to the buffer
        """
        if self.from_file is None:
            while not self.stopped:
                input = self.stream.read(2 * self.hop)
                self.buffer.put(input)
                if self.record_to_file:
                    self.f.writeframesraw(input)
        else:
            while not self.stopped:
                input = self.wf.readframes(2 * self.hop)
                self.buffer.put(input)


    def stop(self):
        """Stop thread and close stream
        """
        self.stopped = True
        time.sleep(self.hop / self.sample_rate)
        self.stream.stop_stream()
        self.stream.close()
        if self.record_to_file:
            self.f.writeframes(b'')
            self.f.close()

class Audio:
    """Class to record and play data

    Application can create an instance of this class and pass it to models.
    The application can use it's methods (or methods as callbacks) to fetch
    some more data for next iteration of a model.
    Class should utilize sliding buffer, so user needs only to read that buffer,
    STFT it and the processing is set up.

    Args:
        buffer_size (int, optional): number of samples in a single output frame
        buffer_hop (int, optional): number of samples that gets removed from a buffer on a single read
        sample_rate (int, optional): sample rate [Hz] of recording
        n_in_channels (int, optional): number of input channels
        n_out_channels (int, optional): number of output channels
        input_device_id (int, optional): Index of input Device to use
        output_device_id (int, optional): Index of input Device to use
        input_from_file (str, optional): Path to the file from which read input,
            if not provided, than it will be get input from input audio device
        save_input (bool, optional): Save recorded input also to the file
            stored in 'records/inputs/', default set to False
        save_output (bool, optional): Save played output also to the file
            stored in 'records/outputs/', default set to False
    """

    def __init__(self, buffer_size=1024, buffer_hop=128, sample_rate=16000,
                 n_in_channels=6, n_out_channels=1, input_device_id=None,
                 output_device_id=None, input_from_file=None, save_input=False,
                 save_output=False):
        self.buffer_size = buffer_size
        self.buffer_hop = buffer_hop
        self.sample_rate = sample_rate
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.input_from_file = input_from_file
        self.save_input = save_input
        self.save_output = save_output

        self.in_buffer = Queue(maxsize=buffer_size / buffer_hop)
        self.out_buffer = Queue(maxsize=buffer_size / buffer_hop)

        self.p = pyaudio.PyAudio()
        self.in_thread = None
        self.out_thread = None

    def write_to_output(self, arr):
        """Decode values and pass it to the buffer

        Args:
            arr (np.array of shape(n_out_channels, buffer_hop)): Frames to be played
        """
        #assert arr.shape == (self.n_out_channels, self.buffer_hop), 'incorect shape of output'
        interleaved = arr.flatten('F')
        self.out_buffer.put(interleaved.tobytes())

    def get_input(self):
        """Get values from the buffer, encode it and return

        Retruns:
            np.array of the shape (n_in_channels, buffer_hop)
        """
        b = self.in_buffer.get()
        arr = np.fromstring(b, dtype=float)
        arr = np.reshape(arr, (self.n_in_channels, self.buffer_hop), order='F')

        return arr

    def open(self):
        """Create and start threads
        """
        self.in_thread = ReadThread(p=self.p,
                                    buffer=self.in_buffer,
                                    hop=self.buffer_hop,
                                    sample_rate=self.sample_rate,
                                    channels=self.n_in_channels,
                                    id=self.input_device_id,
                                    from_file=self.input_from_file,
                                    record_to_file=self.save_input)
        self.out_thread = PlayThread(p=self.p,
                                     buffer=self.out_buffer,
                                     hop=self.buffer_hop,
                                     sample_rate=self.sample_rate,
                                     channels=self.n_out_channels,
                                     id=self.output_device_id,
                                     record_to_file=self.save_output)
        self.in_thread.start()
        self.out_thread.start()

    def close(self):
        """Close all threads
        """
        self.in_thread.stop()
        self.out_thread.stop()
        self.in_thread = None
        self.out_thread = None
        self.p.terminate()

    def __exit__(self, type, value, tb):
        # TODO: handle exception
        return self.close()


if __name__ == "__main__":
    """
    Simple test of whether the class works - a single-channel loopback recording
    """
    audio = Audio(n_in_channels=1, n_out_channels=1)
    audio.open(input_from_file='records/inputs/Tue Jan 22 10:28:26 2019_out.wav',
               save_output=True)
    for _ in range(1000):
        input = audio.get_input()
        audio.write_to_output(input)
    audio.close()
