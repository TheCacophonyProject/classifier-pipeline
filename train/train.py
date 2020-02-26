import datetime
import os
import pickle

import tensorflow as tf
from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ, Model_CNN
from model_resnet import ResnetModel
from ml_tools.dataset import dataset_db_path


def train_model(run_name, conf, hyper_params):
    """Trains a model with the given hyper parameters.
    """
    run_name = os.path.join("train", run_name)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    datasets_filename = dataset_db_path(conf)
    with open(datasets_filename, "rb") as f:
        dsets = pickle.load(f)
    labels = dsets[0].labels
    if conf.train.model == ResnetModel.MODEL_NAME:
        model = ResnetModel(labels, conf.train)
    elif conf.train.model == ModelCRNN_HQ.MODEL_NAME:
        model = ModelCRNN_HQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    elif conf.train.model == Model_CNN.MODEL_NAME:
        model = Model_CNN(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    else:
        model = ModelCRNN_LQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )

    model.import_dataset(datasets_filename)
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

    print("Training started")
    print("---------------------")
    print("Hyper parameters")
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print("Found {0:.1f}K training examples".format(model.rows / 1000))
    print()

    model.train_model(
        epochs=conf.train.epochs,
        run_name=run_name + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    model.save()
    model.close()

    # this shouldn't be nessesary, but unfortunately my model.close isn't cleaning up everything.
    # I think it's because i'm adding everything to the default graph?
    tf.reset_default_graph()

    return model
