#!/bin/bash

cd /home/nvidia/DASH
python3 GUI.py&
GUIPID=$!
python3 runtime.py -audio_config $PWD/configs/small_hop_input_config.yaml -post_filter_config $PWD/configs/postfilter.yaml -model_config $PWD/configs/null_model.yaml -timeit&
RUNTIMEPID=$!
while kill -0 $GUIPID; do
	if ! kill -0 $RUNTIMEPID; then
		python3 runtime.py -audio_config $PWD/configs/small_hop_input_config.yaml -post_filter_config $PWD/configs/postfilter.yaml -model_config $PWD/configs/null_model.yaml -timeit&
		RUNTIMEPID=$!
	fi
	sleep 3
done
kill $RUNTIMEPID
