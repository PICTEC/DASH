#!/usr/bin/env python3


class Audio:
    """
    Application can create an instance of this class and pass it to models.
    The application can use it's methods (or methods as callbacks) to fetch
    some more data for next iteration of a model.

    """

    def __init__(self, buffer_size=512, buffer_hop=128, sample_rate=16000, n_in_channels=6,
                 n_out_channels=1):
        """
        Class should utilize sliding buffer, so user needs only to read that buffer, STFT it
        and the processing is set up.

        buffer_size - number of samples in a single output frame
        buffer_hop - number of samples that gets removed from a buffer on a single read
        sample_rate - self explanatory
        n_in_channels - number of input channels
        n_out_channels - number of output channels
        """

    def open(self):
        """
        Opens an audio device and sets up a thread that reads the data from audio device
        to a buffer, and sets up a buffer for playing the sound for the user.
        """

    def close(self):
        """
        Closes all audio devices, playing what's left in the play buffer
        """

    def listen(self):
        """
        Reads the top buffer_size samples from recording buffer, removes top buffer_hop samples
        from the read buffer and returns the recorded sound frame to the user
        Returns: [buffer_size, n_in_channels] of type np.float32 (32-bit wav in [-1, 1] range)
        """

    def play(self, output):
        """
        Appends the data to the play buffer, from which the sound driver plays the sound
        Accepts: [*any int*, n_out_channels] of type np.float32 (32-bit wav in [-1, 1] range)
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
