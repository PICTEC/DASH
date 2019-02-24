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
        play (bool, optional): Play output to the speakers
        record_to_file (bool, optional): Save played output also to the file
            stored in 'records/outputs/', default set to False
        record_name (str, optional): Name of the saved file
    """
    def __init__(self, p, buffer, hop, sample_rate, channels, id=None, play=True,
                 record_to_file=False, record_name=None):
        super(PlayThread, self).__init__()

        self.buffer = buffer
        self.hop = hop
        self.sample_rate = sample_rate
        self.channels = channels
        self.daemon = True
        self.stopped = False
        self.record_to_file = record_to_file
        self.play = play

        if play:
            self.stream = p.open(format=pyaudio.paInt16,
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
            file_name = 'records/outputs/' + time.strftime('%y%m%d_%H%M%S') + '_out.wav'
            if record_name is not None:
                file_name = 'records/outputs/' + record_name
            self.f = wave.open(file_name, 'w')
            self.f.setnchannels(channels)
            self.f.setsampwidth(2)
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
                if self.play:
                    self.stream.write(frames=frames)
                if self.record_to_file:
                    self.f.writeframesraw(frames)
            else:
                time.sleep(0)

    def stop(self):
        """Stop thread, play what's left in the buffer and close stream
        """
        self.stopped = True

        while not self.buffer.empty():
            frames = self.buffer.get()
            if self.play:
                self.stream.write(frames)
            if self.record_to_file:
                self.f.writeframesraw(frames)
        if self.play:
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
            self.input_dtype = np.float32
        else:
            self.wf = wave.open(from_file, 'rb')
            if self.wf.getsampwidth() == 2:
                self.input_dtype = np.int16
            elif self.wf.getsampwidth() == 4:
                self.input_dtype = np.float32
            else:
                raise ValueError("Incorrect width of samples in the file")

        if record_to_file:
            try:
                makedirs('records/inputs')
            except:
                pass
            file_name = 'records/inputs/' + time.strftime('%y%m%d_%H%M%S') + '_in.wav'
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
                # t = time.time()
                input = self.stream.read(self.hop)
                self.buffer.put(input)
                if self.record_to_file:
                    self.f.writeframesraw(input)
                # print("ACQ", time.time() - t)
        else:
            while not self.stopped:
                # t = time.time()
                input = self.wf.readframes(self.hop)
                # print("ACQuisition", time.time() - t)
                # t = time.time()
                self.buffer.put(input)
                # print("Putting to Queue ACQ", time.time() - t)

    def stop(self):
        """Stop thread and close stream
        """
        self.stopped = True
        time.sleep(self.hop / self.sample_rate)
        if self.from_file is None:
            self.stream.stop_stream()
            self.stream.close()
        else:
            self.wf.close()
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
        play_output (bool, optional): Play output to the speakers
        save_input (bool, optional): Save recorded input also to the file
            stored in 'records/inputs/', default set to False
        save_output (bool, optional): Save played output also to the file
            stored in 'records/outputs/', default set to False
        record_name (str, optional): Name of the output file
    """

    def __init__(self, buffer_size=1024, buffer_hop=128, sample_rate=16000,
                 n_in_channels=6, n_out_channels=1, input_device_id=None,
                 output_device_id=None, input_from_file=None, play_output=True,
                 save_input=False, save_output=False, record_name=None):
        self.buffer_size = buffer_size
        self.buffer_hop = buffer_hop
        self.sample_rate = sample_rate
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.input_device_id = input_device_id
        self.output_device_id = output_device_id
        self.input_from_file = input_from_file
        self.play_output = play_output
        self.save_input = save_input
        self.save_output = save_output
        self.record_name = record_name

        #self.in_queue = Queue(maxsize=buffer_size / buffer_hop)
        #self.out_queue = Queue(maxsize=buffer_size / buffer_hop)
        self.in_queue = Queue(maxsize=2)
        self.out_queue = Queue(maxsize=2)
        self.buffer = np.zeros((buffer_size, n_in_channels), dtype=np.float32)

        self.in_thread = None
        self.out_thread = None
        self.input_dtype = None

        self._in_arr_flat = np.empty((self.buffer_hop * self.n_in_channels), dtype=np.float32)
        self._in_arr_flat_int = np.empty((self.buffer_hop * self.n_in_channels), dtype=np.int16)
        self._in_arr = np.empty((self.buffer_hop, self.n_in_channels), dtype=np.float32)
        self._in_ret = np.empty((self.buffer_size, self.n_in_channels), dtype=np.float32)

        self._out_arr = np.empty((self.buffer_hop, self.n_out_channels), dtype=np.int16)
        self._out_interleaved = self._out_arr.flatten()

        self.out_power = 0
        self.out_power_i = 0

        self.in_power = 0
        self.in_power_i = 0

    def write_to_output(self, arr):
        """Decode values and pass it to the buffer

        Args:
            arr (np.array of shape(buffer_hop, n_out_channels)): Frames to be played
        """
        assert arr.shape == (self.buffer_hop, self.n_out_channels) or arr.shape == (self.buffer_hop,), 'incorrect shape of the output'
        # power = np.sum(arr**2) / (self.buffer_hop * self.n_out_channels)
        # print('output power: ', power)
        # self.out_power += power
        # self.out_power_i += 1
        self._out_arr = (arr*32768).astype(np.int16)
        self._out_interleaved = self._out_arr.flatten()
        self.out_queue.put(self._out_interleaved.tobytes())

    def get_input(self):
        """Get values from the buffer, encode it and return

        Retruns:
            np.array of the shape (buffer_size, n_in_channels)
        """
        b = self.in_queue.get()
        assert len(b) > 0, 'The recording has ended'
        if self.input_dtype == np.int16:
            self._in_arr_flat_int = np.fromstring(b, dtype=np.int16)
            self._in_arr_flat = self._in_arr_flat_int.astype(np.float32) / 32768
        else:
            self._in_arr_flat = np.fromstring(b, dtype=self.input_dtype)

        try:
            self._in_arr = np.reshape(self._in_arr_flat, (self.buffer_hop, self.n_in_channels))
        except:
            raise RuntimeError('Incorrect shape of the input')
        # power = np.sum(self._in_arr**2) / (self.buffer_hop * self.n_in_channels)
        # print('input power:', power)
        # self.in_power += power
        # self.in_power_i += 1
        self.buffer = np.roll(self.buffer, -self.buffer_hop, axis=0)
        self.buffer[-self.buffer_hop:,:] = self._in_arr
        self._in_ret = np.copy(self.buffer)
        return self._in_ret

    def open(self):
        """Create and start threads
        """
        self.p = pyaudio.PyAudio()
        self.in_thread = ReadThread(p=self.p,
                                    buffer=self.in_queue,
                                    hop=self.buffer_hop,
                                    sample_rate=self.sample_rate,
                                    channels=self.n_in_channels,
                                    id=self.input_device_id,
                                    from_file=self.input_from_file,
                                    record_to_file=self.save_input)
        self.out_thread = PlayThread(p=self.p,
                                     buffer=self.out_queue,
                                     hop=self.buffer_hop,
                                     sample_rate=self.sample_rate,
                                     channels=self.n_out_channels,
                                     id=self.output_device_id,
                                     play=self.play_output,
                                     record_to_file=self.save_output,
                                     record_name=self.record_name)
        self.input_dtype = self.in_thread.input_dtype
        self.in_thread.start()
        self.out_thread.start()

    def __enter__(self):
        self.open()
        return self

    def close(self):
        """Close all threads
        """
        self.in_thread.stop()
        self.out_thread.stop()
        self.in_thread = None
        self.out_thread = None
        self.p.terminate()

        # print('Mean input power:', self.in_power / self.in_power_i)
        # print('Mean output power:', self.out_power / self.out_power_i)

    def __exit__(self, type, value, tb):
        # TODO: handle exception
        return self.close()


if __name__ == "__main__":
    """
    Simple test of whether the class works - a single-channel loopback recording
    """
    audio = Audio(n_in_channels=2, n_out_channels=1)
    audio.open()
    for _ in range(1000):
        input = audio.get_input()
        audio.write_to_output(input[:128,1])
    audio.close()
