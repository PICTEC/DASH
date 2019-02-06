class GUI:
    """
    Interface class for graphical user interface for our demonstration
    """

    def start(self):
        """
        Gui shouldn't be enabled by default and should be started with this method

        Once started it shouldn't be disableable unless app has to be stopped
        """

    def put_localization(self, points):
        """
        Draws points of our source computed localization. Points is a list
        and may contain from zero to several (let's say up to four) sources.
        Most common use case will be one or no sources. The method draws the current
        localisation and also keeps a history of several previous localisations of the same
        source. The points from former timesteps should be more and more transparent.
        """

    def put_input_spectrogram(self, bin):
        """
        This method takes one FFT frame and pushes it into buffer which is plotted on screen.
        The buffer should contain log-power STFT rather than plain amplitude or complex spectra.
        Palette should match PICTEC colours.

        Let's assume the input is single channel of shape (fft_size,)
        """

    def put_output_spectrogram(self, bin):
        """
        Same as above, but for the other window
        """

    @property
    def inputs(self, value):
        """
        Properties to selection choices will be dictionaries of form {"WhatIsToBeDisplayed": "SomeInternalKey"}

        GUI should run with uninitialized selection boxes in which case they should do nothing.
        """

    @property
    def outputs(self, value):
        pass

    @property
    def models(self, value):
        pass

    @property
    def postfilters(self, value):
        pass

    @property
    def on_input_change(self, callback):
        """
        Those properties should register callbacks that react to changes to controls in the GUI
        Each callback will probably be linked to some function in runtime.py

        Should pass as an argument internal key associated with the chosen input option

        GUI should run as fine without registered callbacks (e.g. set to None) or with only some
        of them registered. In basic tests we should be able to show plots and disable all possibilities
        of changing the pipeline. Changing the pipeline should be possible in future.
        """

    @property
    def on_output_change(self, callback):
        """
        Should pass as an argument value associated with the chosen output option
        """

    @property
    def on_model_change(self, callback):
        """
        Should pass as an argument value associated with the chosen model
        """

    @property
    def on_postfilter_change(self, callback):
        """
        Should pass as an argument value associated with the postfilter
        """
