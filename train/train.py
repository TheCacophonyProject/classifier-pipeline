import logging
from ml_tools.kerasmodel import KerasModel
import pickle
import os
import faulthandler
from ml_tools.logs import init_logging

faulthandler.enable()


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
    run_name, conf, hyper_params, weights=None, grid_search=None, ignore=None
):
    """Trains a model with the given hyper parameters."""
    init_logging()
    model = KerasModel(train_config=conf.train, labels=conf.labels)

    logging.info("Importing datasets from %s ", conf.tracks_folder)
    model.load_training_meta(conf.tracks_folder)

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
    try:
        model.train_model_tfrecords(
            epochs=conf.train.epochs,
            run_name=run_name + "_" + "TEST",
            base_dir=conf.tracks_folder,
            weights=weights,
            resample=True,
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
