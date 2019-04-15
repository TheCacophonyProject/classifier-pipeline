"""
Author Matthew Aitchison
Date July 2018

This script handles the training of models.  It works in two different modes

Single model training
-----------------------

run the script with
>> train.py [model_name]

to train a single model.  Right now the hyper-parameters need to be in the script before running

Search mode
-----------------------
Many models will be trained across a set of predified hyper parameters.  This can take days, but gives useful information.
I have to dictionaryies, a SHORT_SERACH_PARAMS which has just a few interesting parameters to try, and FULL_SEARCH_PARAMS
which is more comprehensive.

The results of the search are stored in the text file RESULTS_FILENAME, and jobs that have already been processed will
not be redone (however cancelling partway through a job will cause it restart from the start of the job).

Checking the results
-----------------------
All the training results are stored in tensorboard.  To assess the training run tensorboard from the log directory.

"""

import matplotlib

matplotlib.use("Agg")  # enable canvas drawing

import logging
import pickle
import os
import datetime
import argparse
import ast

import tensorflow as tf

from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ
from ml_tools.config import Config
from ml_tools.dataset import dataset_db_path
from ml_tools.logs import init_logging

# name of the file to write results to.
RESULTS_FILENAME = "batch training results.txt"

# this is a good list for a full search, but will take a long time to run (days)
FULL_SEARCH_PARAMS = {
    "batch_size": [1, 2, 4, 8, 16, 32, 64],
    "learning_rate": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "l2_reg": [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    "label_smoothing": [0, 0.01, 0.05, 0.1, 0.2],
    "keep_prob": [0, 0.1, 0.2, 0.5, 0.8, 1.0],
    "batch_norm": [True, False],
    "lstm_units": [64, 128, 256, 512],
    "enable_flow": [True, False],
    "augmentation": [True, False],
    "thermal_threshold": [-100, -20, -10, 0, 10, 20, 100],
    # these runs will be identical in their parameters, it gives a feel for the varience during training.
    "identical": [1, 2, 3, 4, 5],
}

# this checks just the important parameters, and only around areas that are likely to work well.
# I've also excluded the default values as these do not need to be tested again.
SHORT_SEARCH_PARAMS = {
    "batch_size": [8, 32],
    "l2_reg": [1e-2, 1e-3, 1e-4],
    "label_smoothing": [0, 0.05, 0.2],
    "keep_prob": [0.1, 0.4, 0.6, 1.0],
    "batch_norm": [False],
    "lstm_units": [128, 512],
    "enable_flow": [False],
    "augmentation": [False],
    "thermal_threshold": [-100, -20, -10, 0, 20, 100],
}


def train_model(config, rum_name, epochs=30.0, **kwargs):
    """Trains a model with the given hyper parameters. """
    logging.basicConfig(level=0)  # FIXME: calling basicConfig twice is poor
    tf.logging.set_verbosity(3)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    dataset_name = dataset_db_path(config)
    dsets = pickle.load(open(dataset_name, "rb"))
    labels = dsets[0].labels

    model = ModelCRNN_LQ(labels=len(labels), **kwargs)

    model.import_dataset(dataset_name)
    model.log_dir = config.logs_folder

    # display the data set summary
    print("Training on labels", labels)
    print()
    print(
        "{:<20} {:<20} {:<20} {:<20} (segments/tracks/bins/weight)".format(
            "label", "train", "validation", "test"
        )
    )
    for label in labels:
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.datasets.validation.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
            )
        )
    print()

    for dataset in dsets:
        print(dataset.labels)

    print("Training started")
    print("---------------------")
    print("Hyper parameters")
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print("Found {0:.1f}K training examples".format(model.rows / 1000))
    print()
    model.train_model(
        epochs=epochs,
        run_name=rum_name + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    model.save()
    model.close()

    # this shouldn't be nessesary, but unfortunately my model.close isn't cleaning up everything.
    # I think it's because i'm adding everything to the default graph?
    tf.reset_default_graph()

    return model


def has_job(job_name):
    """ Returns if this job has been processed before or not. """
    f = open(RESULTS_FILENAME, "r")
    for line in f:
        words = line.split(",")
        job = words[0] if len(words) >= 1 else ""
        if job == job_name:
            f.close()
            return True
    f.close()
    return False


def log_job_complete(job_name, score, params=None, values=None):

    """ Log reference to job being complete
    :param job_name The name of the job
    :param score The final evaluation score
    :param params [optional] A list of the parameters used to train this model
    :param values [optional] A corresponding list of the parameter values used to train this model

    """
    f = open(RESULTS_FILENAME, "a")
    f.write(
        "{}, {}, {}, {}\n".format(
            job_name,
            str(score),
            params if params is not None else "",
            values if values is not None else "",
        )
    )
    f.close()


def run_job(job_name, **kwargs):
    """ Run a job with given hyper parameters, and log its results. """

    # check if we have done this job...
    if has_job(job_name):
        return

    print("-" * 60)
    print("Processing", job_name)
    print("-" * 60)

    model = train_model("search/" + job_name, **kwargs)

    log_job_complete(
        job_name, model.eval_score, list(kwargs.keys()), list(kwargs.values())
    )


def axis_search():
    """
    Evaluate each hyper-parameter individually against a reference.

    The idea here is to assess each parameter individually while holding all other parameters at their default.
    For optimal results this will need to be done multiple times, each time updating the defaults to their optimal
    values.

    """

    if not os.path.exists(RESULTS_FILENAME):
        open(RESULTS_FILENAME, "w").close()

    # run the reference job with default params
    run_job("reference")

    # go through each job and process it.  Would be nice to be able to start a job and pick up where we left off.
    for param_name, param_values in SHORT_SEARCH_PARAMS.items():
        # build the job
        for param_value in param_values:
            job_name = param_name + "=" + str(param_value)
            args = {param_name: param_value}
            run_job(job_name, **args)


def main():
    init_logging()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        default="unnammed",
        help='Name of training job, use "search" for hyper parameter search',
    )
    parser.add_argument(
        "-e", "--epochs", default="30", help="Number of epochs to train for"
    )
    parser.add_argument("-p", "--params", default="{}", help="model parameters")
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()
    config = Config.load_from_file(args.config_file)

    args = parser.parse_args()

    if args.name == "search":
        print("Performing hyper parameter search.")
        axis_search()
    else:
        # literal eval should be safe here.
        model_args = ast.literal_eval(args.params)
        train_model(config, "training/" + args.name, **model_args)


if __name__ == "__main__":
    # execute only if run as a script
    main()
