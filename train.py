"""This script handles the training of models. It works in two different modes.

# Single model training

run the script with
$ train.py [model_name]

to train a single model. Hyper-parameters are read from classifier.yaml.

# Search mode

Many models will be trained across a set of predified hyper
parameters.  This can take days, but gives useful information.  I have
two dictionaries, a SHORT_SEARCH_PARAMS which has just a few
interesting parameters to try, and FULL_SEARCH_PARAMS which is more
comprehensive.

The results of the search are stored in "search_results.txt" in the
"train" subdirectory of `base_data_folder`. Jobs that have already
been processed will not be redone (however cancelling partway through
a job will cause it restart from the start of the job).

# Checking the results

All the training results are stored in tensorboard.  To assess the
training run tensorboard from the log directory.

"""

import argparse
import os

import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # enable canvas drawing

from ml_tools.logs import init_logging
from config.config import Config
from train.train import train_model
from train.search import axis_search


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "name",
        default="unnammed",
        help='Name of training job, use "search" for hyper-parameter search',
    )
    args = parser.parse_args()
    return Config.load_from_file(args.config_file), args.name


def main():
    conf, job_name = load_config()

    init_logging()
    tf.logging.set_verbosity(3)

    os.makedirs(conf.train.train_dir, exist_ok=True)

    if job_name == "search":
        axis_search(conf)
    else:
        train_model(job_name, conf, conf.train.hyper_params)


if __name__ == "__main__":
    main()
