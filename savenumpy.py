import argparse
import sys
import os
from ml_tools.dataset import dataset_db_path, Dataset, TrackChannels
import joblib
import logging
import numpy as np
from ml_tools import tools
from config.config import Config


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    args = parser.parse_args()
    return args


def init_logging(timestamps=False):
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(levelname)7s %(message)s"
    if timestamps:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def save_numpy(dataset, file):
    logging.info("Writing %s to %s", dataset.name, file)
    track_indexes = {}
    dataset.numpy_file = f"{file}.npy"
    with open(f"{file}.npy", "wb") as f:
        for track in dataset.tracks:
            track_frames = {}
            track_indexes[track.unique_id] = track_frames
            background = dataset.db.get_clip_background(track.clip_id)

            frames = dataset.db.get_track(
                track.clip_id,
                track.track_id,
                original=False,
            )
            f.tell()
            for frame in frames:
                region = track.track_bounds[frame.frame_number]
                region = tools.Rectangle.from_ltrb(*region)
                cropped = region.subimage(background)
                frame.filtered = frame.thermal - cropped
                frame.flow = np.float32(frame.flow)
                if frame.flow is not None and frame.flow_clipped:
                    frame.flow *= 1.0 / 256.0
                    frame.flow_clipped = False

                frame_info = {}
                track_frames[frame.frame_number] = frame_info

                frame_info[TrackChannels.thermal] = f.tell()
                np.save(f, frame.thermal)
                frame_info[TrackChannels.filtered] = f.tell()
                np.save(f, frame.filtered)
                frame_info[TrackChannels.flow] = f.tell()
                np.save(f, frame.flow)
            track.track_info = track_frames
        f.close()


def read_tracks(dataset, file):

    dataset.recalculate_segments()
    with open(f"{file}.npy", "rb") as f:
        for track in dataset.tracks:
            if track.label != "possum":
                continue
            for segment in track.segments:
                for frame_i in segment.frame_indices:
                    frame_info = track.track_info[frame_i]
                    f.seek(frame_info[TrackChannels.thermal])
                    thermal = np.load(f)

                    f.seek(frame_info[TrackChannels.filtered])
                    filtered = np.load(f)


args = load_args()
init_logging()
config = Config.load_from_file(args.config_file)

dataset_file = dataset_db_path(config)
datasets = joblib.load(open(dataset_file, "rb"))
base_dir = os.path.dirname(dataset_file)
for dataset in datasets:
    save_numpy(dataset, os.path.join(base_dir, dataset.name))
    read_tracks(dataset, os.path.join(base_dir, dataset.name))

joblib.dump(datasets, open(os.path.join(base_dir, "numpydataset.dat"), "wb"))
