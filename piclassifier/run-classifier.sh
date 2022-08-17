#!/bin/bash
source /opt/intel/openvino_2021/bin/setupvars.sh
/home/pi/.venv/classifier/bin/python3 /home/pi/classifier-pipeline/piclassify.py --ir
