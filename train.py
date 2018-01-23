"""
Author Matthew Aitchison
Date Jan 2018

Script to train models for classifying animals from thermal footage.
"""

import matplotlib
matplotlib.use('Agg') # enable canvas drawing

import logging
import pickle
import os
import datetime

import tensorflow as tf

from model_crnn import ModelCRNN
from model_crnn_lq import ModelCRNN_LQ
from ml_tools.dataset import Dataset

# folder to put tensor board logs into
LOG_FOLDER = "c:/cac/test_robin/"
# dataset folder to use
DATASET_FOLDER = "c:/cac/robin"

EXCLUDE_LABELS = []

def main():

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    dataset_name = os.path.join(DATASET_FOLDER, 'datasets.dat')
    dsets = pickle.load(open(dataset_name,'rb'))
    labels = dsets[0].labels

    model = ModelCRNN_LQ(labels=len(labels))
    model.import_dataset(dataset_name)
    model.log_dir = LOG_FOLDER

    for dataset in [model.datasets.train, model.datasets.validation, model.datasets.test]:
        dataset.flow_mode = Dataset.FM_NONE

    # display the data set summary
    print("Training on labels: ",labels)
    for label in labels:
        print("{:<20} {:<20} {:<20} {:<20}".format(
            label,
            "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.validation.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
        ))
    print()

    for dataset in dsets:
        print(dataset.labels)

    try:
        print("Training started")
        print()
        print('Hyper parameters')
        print(model.hyperparams_string)
        print()
        print("{0:.1f}K training examples".format(model.rows / 1000))
        model.train_model(epochs=30, run_name='optical flow/LQ/thermal/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        model.save()
    finally:
        model.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()