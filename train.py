"""
Author Matthew Aitchison
Date Jan 2018

Script to train models for classifying animals from thermal footage.
"""

import logging

import tensorflow as tf

from model_crnn import Model_CRNN

# folder to put tensorboard logs into
LOG_FOLDER = "c:/cac/test_robin/"
# dataset folder to use
DATASET_FOLDER = "c:/cac/robin"

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

    model = Model_CRNN(labels=5)
    model.import_dataset(DATASET_FOLDER, force_normalisation_constants=normalisation_constants)
    model.log_dir = LOG_FOLDER

    try:
        print("Training started")
        print()
        print('Hyper parameters')
        print(model.hyperparams_string)
        print()
        print("{0:.1f}K training examples".format(model.rows / 1000))
        model.train_model(epochs=1, run_name='new version')
        model.save_model()
    finally:
        model.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()