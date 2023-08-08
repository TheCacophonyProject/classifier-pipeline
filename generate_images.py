# Script to extra images from datasets with meta_data #
# This can be used to give to people who dont want to work with the h5py db
from multiprocessing import Process, Queue
from PIL import Image
import numpy as np
import pickle
import argparse
import logging
import os
import sys
from config.config import Config
from ml_tools.imageprocessing import normalize
import pickle


def save_job(queue, dataset, folder, labels_dir):
    i = 0
    while True:
        i += 1
        track = queue.get()
        if track == "DONE":
            break
        else:
            save_track(dataset, track, folder, labels_dir)
        if i % 50 == 0:
            logging.info("%s jobs left", queue.qsize())


def save_track(dataset, track, folder, labels_dir, ext="png"):
    track_data = dataset.fetch_track(track, original=True)
    clip_meta = dataset.db.get_clip_meta(track.clip_id)

    clip_dir = os.path.join(folder, str(track.clip_id))
    if not os.path.isdir(clip_dir):
        logging.debug("Creating %s", clip_dir)
        os.mkdir(clip_dir)

    background = dataset.db.get_clip_background(track.clip_id)
    if background is not None:
        background, _ = normalize(background, new_max=255)

        img = Image.fromarray(np.uint8(background))
        filename = "{}-background.{}".format(track.clip_id, ext)
        logging.debug("Saving %s", os.path.join(clip_dir, filename))
        img.save(os.path.join(clip_dir, filename))

    for i, frame in enumerate(track_data):
        thermal = frame.thermal
        normed, _ = normalize(thermal, new_max=255)
        img = Image.fromarray(np.uint8(normed))
        filename = "{}-{}.{}".format(track.clip_id, i + track.start_frame, ext)
        logging.debug("Saving %s", os.path.join(clip_dir, filename))

        img.save(os.path.join(clip_dir, filename))
    save_metadata(clip_meta, track, labels_dir)


def save_metadata(clip_meta, track, folder):
    meta_dir = os.path.join(folder, track.label)
    if not os.path.isdir(meta_dir):
        logging.debug("Creating %s", meta_dir)
        os.mkdir(meta_dir)
    filename = "{}-{}.txt".format(track.clip_id, track.track_id)
    fullpath = os.path.join(meta_dir, filename)
    with open(fullpath, "w") as f:
        f.write(track.toJSON(clip_meta))


def save_all(dataset, worker_threads, folder):
    if not os.path.isdir(folder):
        logging.debug("Creating %s", folder)
        os.mkdir(folder)
    labels_dir = os.path.join(folder + "-labels")
    if not os.path.isdir(labels_dir):
        logging.debug("Creating %s", labels_dir)
        os.mkdir(labels_dir)
    job_queue = Queue()
    processes = []
    for i in range(max(1, 4)):
        p = Process(
            target=save_job,
            args=(job_queue, dataset, folder, labels_dir),
        )
        processes.append(p)
        p.start()

    for track in dataset.tracks:
        job_queue.put(track)

    logging.debug("Processing %d", len(dataset.tracks))
    for i in range(len(processes)):
        job_queue.put("DONE")
    for process in processes:
        process.join()


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--base-folder", help="Base folder to save, train test and val"
    )

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


args = load_args()
init_logging()
config = Config.load_from_file(args.config_file)
datasets = ["train", "validation", "test"]

base_dir = "."
if args.base_folder:
    base_dir = args.base_folder
if not os.path.isdir(base_dir):
    logging.debug("Creating %s", base_dir)
    os.mkdir(base_dir)
for i, name in enumerate(datasets):
    dataset = pickle.load(open(f"{os.path.join(config.tracks_folder, name)}.dat", "rb"))
    dataset.load_db()
    save_all(dataset, config.worker_threads, os.path.join(base_dir, dataset.name))
