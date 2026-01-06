#!/bin/bash
set -e
set -x
config=$2
echo "Saving into $1 with config $2"
month_ago=$(python3 rebuildDate.py $1)
echo $month_ago
cd ../../cptv-download
db_file=$(python3 backup-download.py ./dbs)
sudo -u postgres pg_restore --clean -d cacodb $db_file
cd ../../classifier-pipeline/src
python3 ../../cptv-download/cptv-download-direct.py --start-date "$month_ago" "$1"
echo "Downloading into $1"
python3 build.py -c $config --ext ".cptv" $1
dt=$(date '+%d%m%Y-%H%M%S');
python3 train.py -c $config $dt 
