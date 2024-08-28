#!/bin/bash
set -e
set -x
conda init bash
conda activate tf
config="classifier-thermal.yaml"
echo "Saving into $1"
month_ago=$(python3 rebuildDate.py $1)
echo $month_ago
python3 ../../cptv-download/cptv-download.py -l 0 -i 'poor tracking' -i 'untagged' -i 'part' -i 'untagged-by-humans' -i 'unknown' -i 'unidentified' -m 'human-tagged' --start-date "$month_ago" "$1" useremail@email.com userpassword
echo "Downloading into $1"
python3 build.py -c $config --ext ".cptv" $1
dt=$(date '+%d%m%Y-%H%M%S');
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/cp/miniconda3/envs/tf/lib/
python3 train.py -c $config $dt 