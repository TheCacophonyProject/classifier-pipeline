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

from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ

# folder to put tensor board logs into
LOG_FOLDER = "c:/cac/logs/"
# dataset folder to use
DATASET_FOLDER = "c:/cac/datasets/fantail"

EXCLUDE_LABELS = []

def main():

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    dataset_name = os.path.join(DATASET_FOLDER, 'datasets.dat')
    dsets = pickle.load(open(dataset_name,'rb'))
    labels = dsets[0].labels

    model = ModelCRNN_HQ(labels=len(labels), enable_flow=True)
    model.import_dataset(dataset_name)
    model.log_dir = LOG_FOLDER

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

    print("Training started")
    print()
    print('Hyper parameters')
    print(model.hyperparams_string)
    print()
    print("{0:.1f}K training examples".format(model.rows / 1000))
    model.train_model(epochs=30, run_name='V30/improved model delta frames/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.save()
    model.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()