import logging
from ml_tools.kerasmodel import KerasModel, grid_search
import pickle
import os
from ml_tools.logs import init_logging
from pathlib import Path
import absl.logging
import sys

# tensorflow stealing my log handler
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        root_logger.removeHandler(handler)


def remove_fp_segments(datasets, ignore_file):
    # testing removing some automatically selected bad clips
    unique_ids = ignore_clips(ignore_file)
    print("ignore ids", unique_ids)
    for dataset in datasets:
        delete_me = []
        for segment in dataset.segments:
            if segment.unique_track_id in unique_ids:
                dataset.segments_by_label[segment.label].remove(segment)
                del dataset.segments_by_id[segment.id]
                delete_me.append(segment)
                print("deleting segment", segment.unique_track_id)
        for delete in delete_me:
            try:
                datset.remove_track(delete.track_id)
            except:
                pass
            dataset.segments.remove(delete)
            print("delete track", delete.track_id, " from", dataset.name)
        dataset.rebuild_cdf()


def train_model(
    run_name,
    conf,
    hyper_params,
    weights=None,
    do_grid_search=None,
    ignore=None,
    epochs=None,
):
    init_logging()
    """Trains a model with the given hyper parameters."""
    data_dir = Path(conf.base_folder) / "training-data"
    model = KerasModel(
        train_config=conf.train,
        labels=conf.labels,
        data_dir=data_dir,
    )

    logging.info("Importing datasets from %s ", data_dir)
    model.load_training_meta(data_dir)

    if do_grid_search:
        print("Searching hparams")
        grid_search(model)
        model.close()
        return
    print()
    print("Training started")
    print("---------------------")
    print("Hyper parameters")
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print("EPOCHS ", epochs)
    try:
        model.train_model(
            epochs=conf.train.epochs if epochs is None else epochs,
            run_name=run_name,
            weights=weights,
            resample=False,
            rebalance=False,
        )
    except KeyboardInterrupt:
        pass
    except:
        logging.error("Exited with error ", exc_info=True)
    model.close()


def ignore_clips(file_path):
    ignore_clips = []
    with open(file_path) as stream:
        for line in stream:
            if line.strip() == "" or line[0] != "[":
                continue
            try:
                line = line.replace("[", "")
                line = line.replace("]", "")
                clips = line.split(",")

                for clip in clips:
                    clip = clip.replace("'", "")
                    ignore_clips.append(clip.strip())
            except:
                logging.warn(
                    "Could not parse clip_id %s from %s",
                    line,
                    file_path,
                )
    return ignore_clips
