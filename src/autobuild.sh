#!/bin/sh

set -e
set -x
config="classifier-thermal.yaml"
month_ago=$(python3 rebuildDate.py -c $config)
echo $month_ago
python3 ../../cptv-download/cptv-download.py -l 0 -i 'poor tracking' -i 'untagged' -i 'part' -i 'untagged-by-humans' -i 'unknown' -i 'unidentified' -m 'human-tagged' --start-date "$month_ago" "../clips$month_ago" useremail@email.com userpassword
echo "Downloading into ../clips$month_ago"
python3 load.py -target "../clips$month_ago"  -c $config
python3 build.py -c $config
