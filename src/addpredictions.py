import argparse
from ml_tools.logs import init_logging
from config.config import Config
from pathlib import Path
from ml_tools.rawdb import RawDatabase
import logging
from classify.trackprediction import TrackPrediction
from config.buildconfig import BuildConfig
from ml_tools.dataset import Dataset, filter_clip, filter_track
from ml_tools.kerasmodel import KerasModel
from multiprocessing import Pool
import numpy as np
import json
from ml_tools.tools import CustomJSONEncoder
import cv2
from ml_tools.forestmodel import ForestModel


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
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("-e", action="count", help="Path to config file to use")

    args = parser.parse_args()

    return args


def main():
    init_logging()
    args = parse_args()
    config = Config.load_from_file(args.config_file)
    cptv_dir = Path(args.cptv_dir)
    if args.e:
        evaluate_file(cptv_dir)
        return
    # model = KerasModel(train_config=config.train)
    # model.load_model(args.model, weights=args.weights)
    model = ForestModel(args.model)

    evaluate_dir(model, cptv_dir)


def evaluate_file(cptv_file):
    clip_db = RawDatabase(cptv_file)
    clip_db.load_frames()
    meta_data = clip_db.meta_data
    model_labels = meta_data["fp_model_labels"]
    clip_header = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS)
    save_dir = Path("./test-images")
    save_dir.mkdir(exist_ok=True)
    for track in clip_header.tracks:
        # if track.label
        track_dir = save_dir / f"{track.get_id()}"
        track_dir.mkdir(exist_ok=True)
        fp_predictions = track.fp_frames["predictions"]
        for region in track.regions_by_frame.values():
            frame = clip_db.frames[region.frame_number].thermal
            thermal = region.subimage(frame)
            fp_pred = [p for p in fp_predictions if p["frames"] == region.frame_number]
            if len(fp_pred) == 0:
                logging.info("No pred for %s", region.frame_number)
                continue
            fp_pred = fp_pred[0]

            best_arg = np.argmax(fp_pred["prediction"])
            best_conf = fp_pred["prediction"][best_arg]
            best_conf = best_conf
            best_lbl = model_labels[best_arg]
            logging.info(
                "For frame %s label is %s got pred %s ",
                region.frame_number,
                best_lbl,
                track.label,
            )
            lbl_dir = track_dir / best_lbl
            lbl_dir.mkdir(exist_ok=True)
            thermal = (
                255
                * (np.float32(thermal) - np.amin(thermal))
                / (np.amax(thermal) - np.amin(thermal))
            )
            thermal = np.uint8(thermal)
            cv2.imwrite(
                str(lbl_dir / f"{region.frame_number}-{best_conf}.png"), thermal
            )


worker_model = None


def init_worker(model):
    global worker_model
    worker_model = model


def evaluate_dir(
    model,
    dir,
):
    logging.info("Evaluating cptv files in %s ", dir)

    all_files = list(dir.glob(f"**/*.cptv"))
    total_files = len(all_files)
    index = 0
    load_size = 1000
    while True:
        if index >= total_files:
            break
        files = all_files[index : index + load_size]
        index+= load_size
        with Pool(
            processes=8,
            initializer=init_worker,
            initargs=(model,),
        ) as pool:
            for clip_data in pool.imap_unordered(load_clip_data, files):
                if clip_data is None:
                    continue
                cptv_file = clip_data[0]
                meta_data = clip_data[1]
                clip_data = clip_data[2]
                meta_data["fp_model_labels"] = model.labels
                print(model.version)
                meta_data["fp_model_version"] = float(model.version)
                for data in clip_data:
                    track_id = data[0]
                    label = data[1]
                    preprocessed = data[3]
                    masses = data[4]
                    if model.TYPE == "RandomForest":
                        output = preprocessed
                    else:
                        output = model.predict(preprocessed)
                    prediction = TrackPrediction(data[0], model.labels)
                    prediction.classified_clip(output, output, data[2],masses)
                    prediction.predictions = sorted(
                        prediction.predictions, key=lambda p: p.frames
                    )  # sort by age
                    logging.info("")
                    for track in meta_data["Tracks"]:
                        if track["id"] == track_id:
                            track["fp_model_predictions"] = prediction.get_metadata()
                            break
                meta_file = cptv_file.with_suffix(".txt")
                logging.info("Saving metadata %s", meta_file)
                with meta_file.open("w") as t:
                    # add in some metadata stats
                    json.dump(meta_data, t, indent=4, cls=CustomJSONEncoder)


def load_clip_data(cptv_file):
    # for clip in dataset.clips:
    reason = {}
    clip_db = RawDatabase(cptv_file)
    # tracks = meta_data.get("Tracks", [])
    try:
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
            and track.fp_frames is None
        ]
        logging.info("Looking at cptv %s", cptv_file)
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
                if worker_model.TYPE == "RandomForest":
                    predictions, frames, masses = worker_model.predict_track(clip_db,track)
                    data.append(
                        (
                            track.get_id(),
                            track.label,
                            frames,
                            predictions,
                            masses,
                        )
                    )
                else:
                    frames, preprocessed, masses = worker_model.preprocess(
                        clip_db,
                        track,
                        frames_per_classify=worker_model.params.square_width,
                        dont_filter=True,
                    )

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
        logging.error("Problem loading ", exc_info=True)
        return None


if __name__ == "__main__":
    main()
