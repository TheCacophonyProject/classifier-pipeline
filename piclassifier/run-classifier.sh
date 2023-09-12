#!/bin/bash
source /opt/intel/openvino_2022/setupvars.sh
/home/pi/classifier/bin/python3 /home/pi/classifier-pipeline/piclassify.py -c /home/pi/classifier-pipeline/pi-classifier.yaml
