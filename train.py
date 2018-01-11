"""
Author Matthew Aitchison
Date Jan 2018

Script to train models for classifying animals from thermal footage.
"""

import logging
import pickle
import os

import tensorflow as tf

from model_crnn import ModelCRNN

# folder to put tensorboard logs into
LOG_FOLDER = "c:/cac/test_robin/"
# dataset folder to use
DATASET_FOLDER = "c:/cac/robin"

EXCLUDE_LABELS = ['human']

def main():

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    # normalisation constants are generated during the dataset build step, however i've found it convenient
    # to set them by hand which means overwriting them here.  Mostly so that they don't change every time I rebuild
    # the dataset.
    normalisation_constants = [
        [3200, 180],
        [5.5, 25],
        [0, 0.11],
        [0, 0.10],
        [0, 1]
    ]

    dsets = pickle.load(open(os.path.join(DATASET_FOLDER, 'datasets.dat'),'rb'))
    labels = dsets[0].labels

    model = ModelCRNN(labels=len(labels))
    model.import_dataset(DATASET_FOLDER, force_normalisation_constants=normalisation_constants, ignore_labels=EXCLUDE_LABELS)
    model.log_dir = LOG_FOLDER

    labels = [label for label in labels if label not in EXCLUDE_LABELS]

    # display the dataset summary
    print("Training on labels: ",labels)
    for label in labels:
        print("{:<20} {:<20} {:<20} {:<20}".format(
            label,
            "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.validation.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
        ))
    print()

    try:
        print("Training started")
        print()
        print('Hyper parameters')
        print(model.hyperparams_string)
        print()
        print("{0:.1f}K training examples".format(model.rows / 1000))
        model.train_model(epochs=30, run_name='thermal/stoatv3')
        model.save_model()
    finally:
        model.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()