import logging
from ml_tools.kerasmodel import KerasModel


def train_model(run_name, conf, hyper_params, weights=None, grid_search=None):
    """Trains a model with the given hyper parameters."""

    model = KerasModel(train_config=conf.train, labels=conf.labels)

    logging.info("Importing datasets from %s ", conf.tracks_folder)
    model.import_dataset(conf.tracks_folder, lbl_p=conf.train.label_probabilities)

    groups = {}
    false_positives = ["false-positive", "insect"]
    for label in model.train_dataset.labels:
        if label not in false_positives:
            groups[label] = [label]

    groups["false-positives"] = false_positives
    model.mapped_labels = groups
    model.regroup()

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

    if grid_search:
        print("Searching hparams")
        test_hparams(model)
        model.close()
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
        for segment in dataset.segments:
            segment.frame_temp_median = None
            segment.regions = None
    try:
        model.train_model(
            epochs=conf.train.epochs, run_name=run_name + "_" + "TEST", weights=weights
        )
    except KeyboardInterrupt:
        pass
    except:
        logging.error("Exited with error ", exc_info=True)
    model.close()
