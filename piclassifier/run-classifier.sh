#!/bin/bash

set -e # Exit if any commands fail

file_path="/opt/intel/openvino_2022/setupvars.sh"
if [ -f "$file_path" ]; then
  source "$file_path"
fi
/home/pi/classifier/bin/python3 /home/pi/classifier-pipeline/piclassify.py -c /home/pi/classifier-pipeline/pi-classifier.yaml
