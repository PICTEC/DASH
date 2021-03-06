from PyQt5 import QtCore, QtWidgets, QtGui, QtWebEngineWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QSize, pyqtSignal

import pyqtgraph as pg
import numpy as np
import paho.mqtt.client as mqtt

pg.setConfigOption('background', '#423b6a')

import time
import json
import struct


class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, frame_width, frame_hop, sample_rate):
        super(SpectrogramWidget, self).__init__()
        self.frame_width = frame_width
        self.hop_per_width = int(frame_width / frame_hop)
        self.setMouseEnabled(False, False)
        self.getPlotItem().setMenuEnabled(False)

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((750, int(frame_width/2+1))) - 255

        pos = np.array([0., 0.5,  1.])
        color = np.array([(255,255,255,0), (19,134,111,255), (255,0,0,255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1, 255)

        self.img.setLookupTable(lut)
        self.img.setLevels([-150,-50])

        freq = np.arange((frame_width/2)+1)/(frame_width/sample_rate)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./sample_rate)*frame_width/self.hop_per_width, yscale)

        self.setLabel('left', 'Frequency', units='Hz')

        self.show()

    def update(self, chunk):
        chunk = 20 * np.log10(chunk)
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = chunk

        self.img.setImage(self.img_array, autoLevels=False)

class LocalizationWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(float)
    def __init__(self):
        super().__init__()

        self.plot = pg.PlotItem()
        self.addItem(self.plot)
        self.heightForWidth = 1

        self.localizations_array = np.zeros((25))
        self.vanishing = np.linspace(255, 1, 25, dtype=int)
        self.pen = []
        for i in range(25):
            self.pen.append(pg.mkPen(color=(19, 134, 111, self.vanishing[i]), width=1.5))

        hor_line = pg.QtGui.QGraphicsLineItem(-1, 0, 1, 0)
        ver_line = pg.QtGui.QGraphicsLineItem(0, -1, 0, 1)
        diag_line1 = pg.QtGui.QGraphicsLineItem(-np.tan(np.pi/4), -np.tan(np.pi/4), np.tan(np.pi/4), np.tan(np.pi/4))
        diag_line2 = pg.QtGui.QGraphicsLineItem(-np.tan(np.pi/4), np.tan(np.pi/4), np.tan(np.pi/4), -np.tan(np.pi/4))
        self.circle = pg.QtGui.QGraphicsEllipseItem(-1, -1, 2, 2)
        self.circle.setPen(pg.mkPen(color=[255,255,255, 255], width=0.5))
        self.plot.addItem(self.circle)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.plot.hideAxis('top')
        self.plot.hideAxis('right')
        self.plot.setXRange(-1.01, 1.01, padding=0)
        self.plot.setYRange(-1.01, 1.01, padding=0)
        self.plot.hideButtons()
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.show()

    def update(self, new_localization):
        self.localizations_array = np.roll(self.localizations_array, 1)
        self.localizations_array[0] = new_localization

        self.plot.clear()
        self.plot.addItem(self.circle)
        for i in range(25):
            ang = self.localizations_array[i]
            line = pg.QtGui.QGraphicsLineItem(0, 0, np.cos(ang), np.sin(ang))
            line.setPen(self.pen[i])
            self.plot.addItem(line)

def input_spectrogram_callback(client, userdata, message):
    userdata.put_input_spectrogram(np.fromstring(message.payload, dtype=np.complex64))

def output_spectrogram_callback(client, userdata, message):
    userdata.put_output_spectrogram(np.fromstring(message.payload, dtype=np.complex64))

def localization_callback(client, userdata, message):
    userdata.put_localization(float(message.payload))

def configuration_callback(client, userdata, message):
    userdata.config_change(message.payload)

class AboutWidgetSignals(QtCore.QObject):
    # SIGNALS
    CLOSE = QtCore.pyqtSignal()

class AboutWidget(QWidget):
    def __init__(self, parent=None):
        super(AboutWidget, self).__init__(parent)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.fillColor = QtGui.QColor(30, 30, 30, 120)
        self.penColor = QtGui.QColor("#333333")

        self.popup_fillColor = QtGui.QColor(240, 240, 240, 255)
        self.popup_penColor = QtGui.QColor(200, 200, 200, 255)

        self.about_window = QtWebEngineWidgets.QWebEngineView(self)
        self.about_window.load(QtCore.QUrl().fromLocalFile('/home/nvidia/DASH/showcase.html'))

        self.about_window.setMinimumSize(1200, 700)
        self.about_window.move(parent.rect().center() - self.about_window.rect().center())

        self.close_btn = QtWidgets.QPushButton(self.about_window)
        self.close_btn.setText("x")
        font = QtGui.QFont()
        font.setPixelSize(18)
        font.setBold(True)
        #font.se
        self.close_btn.setFont(font)
        self.close_btn.setStyleSheet("background-color: rgb(19, 134, 111, 255)")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.move(1170, 0)
        self.close_btn.clicked.connect(self._onclose)

        self.SIGNALS = AboutWidgetSignals()

    def _onclose(self):
        self.SIGNALS.CLOSE.emit()


class GUI(QMainWindow):
    """
    Interface class for graphical user interface for our demonstration
    """
    def __init__(self, width, height):
        super().__init__()
        self.config = {}
        self.screen_width = width
        self.screen_height = height

        self.initUI()

        self.client = mqtt.Client('GUI')
        self.client.user_data_set(self)
        self.client.on_message = input_spectrogram_callback
        self.client.message_callback_add('dash.in', input_spectrogram_callback)
        self.client.message_callback_add('dash.out', output_spectrogram_callback)
        self.client.message_callback_add('dash.pos', localization_callback)
        self.client.message_callback_add('dash.config', configuration_callback)
        self.client.connect('localhost')
        self.client.subscribe('dash.in')
        self.client.subscribe('dash.out')
        self.client.subscribe('dash.pos')
        self.client.subscribe('dash.config')

        self.client.loop_start()

        self.in_spec_i = 5
        self.out_spec_i = 5

        self._popframe = None
        self._popflag = False

    def initUI(self):
        self.setStyleSheet('background-color: #423b6a;')
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.central_layout = QHBoxLayout(self.centralWidget)
        self._create_buttons()
        self.central_layout.addSpacing(50)
        self.central_layout.addLayout(self._control_panel())
        self.central_layout.addSpacing(100)
        self.central_layout.addLayout(self._spectrogram_panel())
        self.central_layout.addLayout(self._localization_panel())
        self.central_layout.addSpacing(10)

        self._connect_signals()

        self.show()
        self.showFullScreen()

    def _create_buttons(self):
        self.button_start = QPushButton('Start', self.centralWidget)
        self.button_start.clicked.connect(self.start)
        self.button_stop = QPushButton('Stop', self.centralWidget)
        self.button_stop.clicked.connect(self.stop)

        self.group_in_out_play = QButtonGroup(self.centralWidget)
        self.radio_in_play = QRadioButton('Play input', self.centralWidget)
        self.radio_in_play.setStyleSheet('color: white')
        self.radio_in_play.clicked.connect(self.play_input)
        self.radio_out_play = QRadioButton('Play output', self.centralWidget)
        self.radio_out_play.setChecked(True)
        self.radio_out_play.setStyleSheet('color: white')
        self.radio_out_play.clicked.connect(self.plat_output)
        self.group_in_out_play.addButton(self.radio_in_play)
        self.group_in_out_play.addButton(self.radio_out_play)

        self.combo_config = QComboBox(self.centralWidget)
        self.combo_config.activated[str].connect(self.publish_config)

        self.button_about = QPushButton('About', self.centralWidget)
        self.button_about.clicked.connect(self._onpopup)

        self.button_exit = QPushButton('Exit', self.centralWidget)
        self.button_exit.clicked.connect(self.close)

    def _control_panel(self):
        control_layout = QVBoxLayout()
        control_start_stop_layout = QHBoxLayout()
        control_inout_play_layout = QHBoxLayout()

        logo = QLabel(self.centralWidget)
        logo_pixmap = QPixmap('bin/logo.png')
        #logo.setPixmap(logo_pixmap.scaled(280, 210))
        logo.setPixmap(logo_pixmap.scaled(int(0.15*self.screen_width), int(0.2*self.screen_height)))

        label_in_out_play = QLabel(self.centralWidget)
        label_in_out_play.setText("<font color='White'> Select what is played </font>")
        label_config = QLabel(self.centralWidget)
        label_config.setText("<font color='White'> Select configuration </font>")

        control_layout.addWidget(logo)
        control_layout.addWidget(self.button_about)
        control_layout.addStretch()

        control_layout.addWidget(label_config)
        control_layout.addWidget(self.combo_config)
        control_layout.addStretch()

        control_layout.addWidget(label_in_out_play)
        control_inout_play_layout.addWidget(self.radio_in_play)
        control_inout_play_layout.addWidget(self.radio_out_play)
        control_layout.addLayout(control_inout_play_layout)
        control_layout.addStretch()

        control_start_stop_layout.addWidget(self.button_start)
        control_start_stop_layout.addWidget(self.button_stop)
        control_layout.addLayout(control_start_stop_layout)
        control_layout.addStretch()

        control_layout.addWidget(self.button_exit)
        control_layout.addSpacing(50)

        return control_layout

    def _spectrogram_panel(self):
        spectrogram_layout = QVBoxLayout()

        label_input_spectrogram = QLabel(self.centralWidget)
        label_input_spectrogram.setText("<font color='White'> Input spectrogram </font")
        label_input_spectrogram.setAlignment(QtCore.Qt.AlignCenter)
        self.input_spectrogram = SpectrogramWidget(512, 128, 16000)
        label_output_spectrogram = QLabel(self.centralWidget)
        label_output_spectrogram.setText("<font color='White'> Output spectrogram </font")
        label_output_spectrogram.setAlignment(QtCore.Qt.AlignCenter)
        self.output_spectrogram = SpectrogramWidget(512, 128, 16000)

        spectrogram_layout.addSpacing(50)
        spectrogram_layout.addWidget(label_input_spectrogram)
        spectrogram_layout.addWidget(self.input_spectrogram)
        spectrogram_layout.addSpacing(20)
        spectrogram_layout.addWidget(label_output_spectrogram)
        spectrogram_layout.addWidget(self.output_spectrogram)
        spectrogram_layout.addSpacing(50)

        return spectrogram_layout

    def _localization_panel(self):
        localization_layout = QVBoxLayout()
        localization_plot_layout = QHBoxLayout()
        localization_logo_layout = QHBoxLayout()

        label_localization = QLabel(self.centralWidget)
        label_localization.setText("<font color='White'> Visit us at... </font")
        label_localization.setAlignment(QtCore.Qt.AlignCenter)
        # self.localization = LocalizationWidget()
        # self.localization.hideAxis('bottom')
        # self.localization.hideAxis('left')
        #self.localization.heightForWidth(1)

        logo = QLabel(self.centralWidget)
        logo_pixmap = QPixmap('bin/logo_big.png')
        #logo.setPixmap(logo_pixmap.scaled(560, 420))
        logo.setPixmap(logo_pixmap.scaled(int(0.3*self.screen_width), int(0.4*self.screen_height)))

        qr_code = QLabel(self.centralWidget)
        qr_code_pixmap = QPixmap('bin/qrcode.png')
        qr_code.setPixmap(qr_code_pixmap.scaled(int(0.3*self.screen_width), int(0.45*self.screen_height)))

        localization_plot_layout.addStretch()
        localization_plot_layout.addWidget(qr_code)
        localization_plot_layout.addStretch()

        localization_logo_layout.addStretch()
        localization_logo_layout.addWidget(logo)
        localization_logo_layout.addStretch()

        # localization_layout.addSpacing(50)
        localization_layout.addWidget(label_localization)
        localization_layout.addLayout(localization_plot_layout)
        #localization_layout.addStretch()
        localization_layout.addLayout(localization_logo_layout)
        #localization_layout.addStretch()

        return localization_layout

    def _connect_signals(self):
        self.input_spectrogram.read_collected.connect(self.input_spectrogram.update)
        self.in_spectrogram_signal = self.input_spectrogram.read_collected
        self.output_spectrogram.read_collected.connect(self.output_spectrogram.update)
        self.out_spectrogram_signal = self.output_spectrogram.read_collected

        # self.localization.read_collected.connect(self.localization.update)
        # self.localization_signal = self.localization.read_collected

    def run(self):
        """
        Gui shouldn't be enabled by default and should be started with this method

        Once started it shouldn't be disableable unless app has to be stopped
        """

    def _onpopup(self):
        self._popframe = AboutWidget(self)
        self._popframe.move(0, 0)
        self._popframe.resize(self.width(), self.height())
        self._popframe.SIGNALS.CLOSE.connect(self._closepopup)
        self._popflag = True
        self._popframe.show()

    def _closepopup(self):
        self._popframe.close()
        self._popflag = False

    def start(self):
        self.client.publish('dash.control', 'START')

    def stop(self):
        self.client.publish('dash.control', 'STOP')

    def play_input(self):
        self.client.publish('dash.control', 'PLAY_IN')

    def plat_output(self):
        self.client.publish('dash.control', 'PLAY_OUT')

    def about(self):
        self.about_window.show()

    def put_localization(self, point):
        """
        Draws points of our source computed localization. Points is a list
        and may contain from zero to several (let's say up to four) sources.
        Most common use case will be one or no sources. The method draws the current
        localisation and also keeps a history of several previous localisations of the same
        source. The points from former timesteps should be more and more transparent.
        """
        # self.localization_signal.emit(point)

    def put_input_spectrogram(self, bin):
        """
        This method takes one FFT frame and pushes it into buffer which is plotted on screen.
        The buffer should contain log-power STFT rather than plain amplitude or complex spectra.
        Palette should match PICTEC colours.

        Let's assume the input is single channel of shape (fft_size,)
        """
        if self.in_spec_i == 5:
            b = np.reshape(bin, (257,8))
            self.in_spectrogram_signal.emit(np.absolute(b[:,0]/16000))
            self.in_spec_i = 0
        else:
            self.in_spec_i += 1

    def put_output_spectrogram(self, bin):
        """
        Same as above, but for the other window
        """
        if self.out_spec_i == 5:
            self.out_spectrogram_signal.emit(np.absolute(bin/16000))
            self.out_spec_i = 0
        else:
            self.out_spec_i += 1

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

    def config_change(self, config):
        default, self.config = json.loads(config)
        IX = -1
        self.combo_config.clear()
        for ix, (k, v) in enumerate(self.config.items()):
            self.combo_config.addItem(k)
            if v == default:
                IX = ix
        if IX != -1:
            self.combo_config.setCurrentIndex(IX)

    def publish_config(self, text):
        self.client.publish('dash.control', 'SWITCH_'+self.config[text])


if __name__ == '__main__':
    app = QtGui.QApplication([])
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    mainWin = GUI(width, height)
    app.exec_()
