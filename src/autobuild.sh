#!/bin/sh

set -e
set -x
config="classifier-thermal.yaml"
# month_ago=$(python3 rebuildDate.py -c $config)
# echo $month_ago
python3 ../../cptv-download/cptv-download.py -l 0 -i 'poor tracking' -i 'untagged' -i 'part' -i 'untagged-by-humans' -i 'unknown' -i 'unidentified' -m 'human-tagged' "/data2/cptv-files" useremail@email.com userpassword
echo "Downloading into /data2/cptv-files
python3 build.py -c $config --ext ".cptv" /data2/cptv-files
