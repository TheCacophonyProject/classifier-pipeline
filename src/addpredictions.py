"""
Author: Giampaolo Ferraro
Date: December 2025

Script used to false-positive or not prediction data into the metadata of cptv files

This metadata can then be used when building a dataset to filter out frames which are tagged as an animal but predicted as false-positive (by this model)
"""

import argparse
from ml_tools.logs import init_logging
from config.config import Config
from pathlib import Path
from ml_tools.rawdb import RawDatabase
import logging
from classify.trackprediction import TrackPrediction
from config.buildconfig import BuildConfig
from ml_tools.dataset import Dataset, filter_clip, filter_track

from multiprocessing import Pool
import numpy as np
import json
from ml_tools.forestmodel import ForestModel
from ml_tools.kerasmodel import KerasModel
from ml_tools.tools import CustomJSONEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="Weights to load into model")
    parser.add_argument(
        "model",
        help="Path to model file to use, will override config model",
    )
    parser.add_argument(
        "cptv_dir",
        help="Evalute directory of cptv files",
    )
    args = parser.parse_args()

    return args


def main():
    init_logging()
    args = parse_args()
    cptv_dir = Path(args.cptv_dir)
    model_path = Path(args.model)
    if model_path.suffix == ".pkl":
        classifier = ForestModel(model_path)
    else:
        classifier = KerasModel()
        classifier.load_model(model_path)

    evaluate_dir(classifier, cptv_dir)


worker_model = None


def init_worker(model):
    global worker_model
    worker_model = model


def evaluate_dir(
    model,
    dir,
    threshold=0.5,
    after_date=None,
):
    logging.info("Evaluating cptv files in %s with threshold %s", dir, threshold)

    files = list(dir.glob(f"**/*cptv"))
    total_files = len(files)
    complete = 0
    with Pool(
        processes=8,
        initializer=init_worker,
        initargs=(model,),
    ) as pool:
        for clip_data in pool.imap_unordered(load_clip_data, files):
            if complete % 100 == 0:
                logging.info("Done %s / %s ", complete, total_files)
            complete += 1
            if clip_data is None:
                continue
            cptv_file = clip_data[0]
            meta_data = clip_data[1]
            clip_data = clip_data[2]
            meta_data["fp_model_labels"] = model.labels
            meta_data["fp_model_version"] = model.version
            for data in clip_data:
                track_id = data[0]
                label = data[1]
                frames = data[2]
                preprocessed = data[3]
                masses = data[4]
                output = model.predict(preprocessed)
                prediction = TrackPrediction(data[0], model.labels)
                prediction.classified_clip(output, 100 * output, frames, masses)
                for track in meta_data["Tracks"]:
                    if track["id"] == track_id:
                        track["fp_model_predictions"] = prediction.get_metadata()
                        break
            meta_file = cptv_file.with_suffix(".txt")
            with meta_file.open("w") as t:
                # add in some metadata stats
                json.dump(meta_data, t, indent=4, cls=CustomJSONEncoder)


def load_clip_data(cptv_file):
    try:
        # for clip in dataset.clips:
        reason = {}
        clip_db = RawDatabase(cptv_file)
        # tracks = meta_data.get("Tracks", [])

        clip = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS)
        if clip is None:
            logging.warn("No clip for %s", cptv_file)
            return None

        if filter_clip(clip, None, None, reason):
            # logging.info("Filtering %s", cptv_file)
            return None
        clip.tracks = [
            track
            for track in clip.tracks
            if not filter_track(track, BuildConfig.EXCLUDED_TAGS, reason)
            # and track.fp_frames is None
        ]
        if len(clip.tracks) == 0:
            logging.info("No tracks after filtering %s", cptv_file)
            return None
        clip_db.load_frames()
        thermal_medians = []
        for f in clip_db.frames:
            thermal_medians.append(np.median(f.thermal))
        thermal_medians = np.uint16(thermal_medians)
        data = []
        for track in clip.tracks:
            try:

                samples = worker_model.frames_for_prediction(clip_db, track)
                frames, preprocessed, masses = worker_model.preprocess(
                    clip_db, track, samples, dont_filter=True
                )
                if preprocessed is None or len(preprocessed) == 0:
                    logging.error("No preprocessed data for %s", track)
                    continue

                data.append(
                    (
                        track.get_id(),
                        track.label,
                        frames,
                        preprocessed,
                        masses,
                    )
                )
            except:
                logging.error("Could not load %s", clip.clip_id, exc_info=True)
        return (cptv_file, clip_db.meta_data, data)
    except:
        logging.error("Could not load %s", cptv_file, exc_info=True)

    return None


if __name__ == "__main__":
    main()
