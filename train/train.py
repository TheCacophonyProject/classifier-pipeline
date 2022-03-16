import logging
from ml_tools.kerasmodel import KerasModel


def train_model(run_name, conf, hyper_params, weights=None, grid_search=None):
    """Trains a model with the given hyper parameters."""
    model = KerasModel(train_config=conf.train, labels=conf.labels)
    model.load_training_meta(conf.tracks_folder)

    logging.info("Importing datasets from %s ", conf.tracks_folder)
    # model.import_dataset(conf.tracks_folder, lbl_p=conf.train.label_probabilities)

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
    # print(
    #     "Found {0:.1f}K training examples".format(
    #         model.train_dataset.sample_count / 1000
    #     )
    # )
    # for dataset in model.datasets.values():
    #     dataset.clear_tracks()
    print("tracks folder", conf.tracks_folder)
    model.train_model_dataset(
        epochs=conf.train.epochs,
        run_name=run_name + "_" + "TEST",
        base_dir=conf.tracks_folder,
        weights=weights,
    )
    model.close()
