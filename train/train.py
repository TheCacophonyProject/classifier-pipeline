import os
import pickle
import time
import logging
from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ, Model_CNN

# from model_resnet import ResnetModel
from ml_tools.kerasmodel import KerasModel


def train_model(run_name, conf, hyper_params, weights=None, grid_search=None):
    """Trains a model with the given hyper parameters."""
    # run_name = os.path.join("train", run_name)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    # datasets_filename = os.path.join(os.path.dirname(datasets_filename), "datasets.dat")
    # if conf.train.model == ResnetModel.MODEL_NAME:
    # model = ResnetModel(labels, conf.train)
    if conf.train.model == ModelCRNN_HQ.MODEL_NAME:
        model = ModelCRNN_HQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    elif conf.train.model == Model_CNN.MODEL_NAME:
        model = Model_CNN(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    elif conf.train.model == "keras":
        model = KerasModel(train_config=conf.train, labels=conf.labels)
    else:
        model = ModelCRNN_LQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    #
    groups = {}
    animals = []
    false_positives = ["false-positive", "insect"]
    # groups["wallaby"] = ["wallaby"]
    #
    # groups["possum"] = ["possum", "cat", "hedgehog"]
    # groups["bird"] = ["bird"]
    # groups["rodent"] = ["rodent"]
    # groups["mustelid"] = ["mustelid"]
    # groups["leporidae"] = ["leporidae"]

    # groups["false-positive"] = ["false-positive", "insect"]
    logging.info("Importing datasets from %s ", conf.tracks_folder)
    model.import_dataset(conf.tracks_folder, lbl_p=conf.train.label_probabilities)
    for label in model.train_dataset.labels:
        if label not in false_positives:
            groups[label] = [label]
        # if label != "wallaby":
        #     animals.append(label)

    groups["false-positives"] = false_positives
    # groups["not"] = animals
    model.mapped_labels = groups
    model.regroup()

    logging.info("IMPORTED")
    # display the data set summary
    print("Training on labels", model.train_dataset.labels)
    print()
    print(
        "{:<20} {:<20} {:<20} {:<20} (segments/frames/tracks/bins)".format(
            "label", "train", "validation", "test"
        )
    )
    for label in model.train_dataset.labels:
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{:.1f}".format(*model.train_dataset.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.validation_dataset.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.test_dataset.get_counts(label)),
            )
        )
    if model.train_dataset.label_mapping:
        print("Mapped labels")
        for label in model.train_dataset.label_mapping.keys():
            print(
                "{} {:<20} {:<20} {:<20} {:<20}".format(
                    label,
                    model.train_dataset.mapped_label(label),
                    "{}/{}/{}/{:.1f}".format(*model.train_dataset.get_counts(label)),
                    "{}/{}/{}/{:.1f}".format(
                        *model.validation_dataset.get_counts(label)
                    ),
                    "{}/{}/{}/{:.1f}".format(*model.test_dataset.get_counts(label)),
                )
            )

    print(weights)
    # if weights:
    #     model.load_weights(weights, meta=False, training=True)
    if grid_search:
        print("Searching hparams")
        model.test_hparams()
        return
    print()
    print("Training started")
    print("---------------------")
    print("Hyper parameters")
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print(
        "Found {0:.1f}K training examples".format(
            model.train_dataset.sample_count / 1000
        )
    )
    for dataset in model.datasets.values():
        dataset.clear_tracks()

    model.train_model(
        epochs=conf.train.epochs, run_name=run_name + "_" + "TEST", weights=weights
    )
    # model.save()
    model.close()
