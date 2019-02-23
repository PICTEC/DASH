#!/bin/bash

cd /home/nvidia/DASH
python3 GUI.py&
GUIPID=$!
python3 runtime.py -audio_config $PWD/configs/large_hop_input_config.yaml -post_filter_config $PWD/configs/null_postfilter.yaml -model_config $PWD/configs/4ch-lstm_mvdr_config.yaml -timeit&
RUNTIMEPID=$!
while kill -0 $GUIPID; do
	if ! kill -0 $RUNTIMEPID; then
		python3 runtime.py -audio_config $PWD/configs/large_hop_input_config.yaml -post_filter_config $PWD/configs/null_postfilter.yaml -model_config $PWD/configs/4ch-lstm_mvdr_config.yaml -timeit&
		RUNTIMEPID=$!
	fi
	sleep 3
done
kill $RUNTIMEPID
