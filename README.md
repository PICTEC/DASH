This version contains reviewed code. For version deployed on Embedded World, checkout [embedded-world branch](https://github.com/PICTEC/DASH/tree/embedded-world). For non-technical information, visit [GitHub page for the project](https://pictec.github.io/DASH).

# DASH - Deep Audio Showcase

This project contains implementations of several speech preprocessing technologies.

### What is featured here?

We provide working implementation of MVDR beamforming with estimation of direction of incidence via main eigenvector of the speech covariance matrix. We also provide training scripts for LSTM masking and denoising autoencoder networks. We provide VAD and GCC-PHAT implementations, altough they were excluded from final presentation for efficiency reasons. We also provide framework for capture and real-time processing of sound using those methods.

### How to use

Application works on top of basic JetPack installation. Python dependencies are listed in requirements. Other system-wide dependencies are not installed automatically - you should provide your own binaries from proper repositories. In particular, _please provide your own Tensorflow installation_ to utilize your available hardware well.

Application is to be installed on `/home/nvidia/DASH`. Launcher script (BeamBox.desktop) can be moved to Desktop for clickable launcher. This demo is build of two components - GUI and proper engine. You may launch them separately by using appropriate scripts. It is advisable to use launcher as calling method for `runtime.py` is verbose.

For good experience you should provide headphones and 8-microphone matrix. The matrix parameters can be found [here](https://pictec.github.io/DASH/matrix_spec.html). Some of the models are rather universal.

Training assets are usually in the form of Jupyter notebooks and are found in `notebooks` directory. Those are also specific to our own environments and may need tuning for your machine.

### Possible extensions

This prototype runs close to the time limit for the frame, however the resource utilization is very small. The networks can easily be parallelized and more channels can be added. We didn't really tested the saturtion of the GPUs with the network, but there is plenty room for optimization of the whole pipeline.

From our profiling, eigenvector decomposition and neural networks are two most expensive operations in the model which should provide highest gains when optimized.

Since CPU time is critical and due to GIL, it is advisable that all possible services using audio from this application are run in separate process. We use [mosquitto broker](https://github.com/eclipse/mosquitto) and [appropriate Python client](https://pypi.org/project/paho-mqtt/) and for fixed-size binary data it seems to be a good choice.

### Useful commands:

Dry run (smoke test for audio):

`python3.6 runtime.py -audio_config configs/local_test_config.yaml -post_filter_config configs/null_postfilter.yaml -model_config configs/null_model.yaml -timeit`

MVDR pipeline:

`python3.6 runtime.py -audio_config configs/local_test_config.yaml -post_filter_config configs/null_postfilter.yaml -model_config configs/null_model.yaml`

### Contributions

If you want to extend this project, notify us via [mail](mailto:pawel.tomasik@pictec.eu).
