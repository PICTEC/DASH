#!/bin/bash

cd /home/nvidia/DASH
python3 GUI.py&
GUIPID=$!
python3 runtime.py&
RUNTIMEPID=$!
while kill -0 $GUIPID; do
	if kill -0 $RUNTIMEPID; then
		python3 runtime.py&
		RUNTIMEPID=$!
	fi
	sleep 3
done
kill $RUNTIMEPID
