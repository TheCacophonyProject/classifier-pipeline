from multiprocessing import Process, Queue

from ml_tools.dataset import Preprocessor
from ml_tools.framedataset import dataset_db_path

import numpy as np
import pickle
from dateutil.parser import parse
import argparse
import logging
import os
import sys
from config.config import Config
from datetime import datetime, timedelta
from ml_tools.datagenerator import preprocess_movement
from ml_tools.kerasmodel import KerasModel
from ml_tools.trackdatabase import TrackDatabase
from classify.trackprediction import Predictions, TrackPrediction

VISIT_INTERVAL = 10 * 60


class CameraResults:
    def __init__(self, camera):
        self.camera = camera
        self.track_results = []
        self.visits = []

    def add_result(self, track, predicted_lbl, prediction_score):
        self.track_results.append((track, predicted_lbl, prediction_score))

    def calc_visits(self):
        start_sorted = sorted(self.track_results, key=lambda res: res[0].start_time)
        last_visit = None
        self.visits = []
        for result in start_sorted:
            if last_visit is None:
                last_visit = Visit(self.camera)
                last_visit.add_track(*result)
                self.visits.append(last_visit)
            else:
                time_diff = last_visit.end_time - result[0].start_time
                if time_diff.total_seconds() < VISIT_INTERVAL:
                    last_visit.add_track(*result)
                else:
                    last_visit = Visit(self.camera)
                    last_visit.add_track(*result)
                    self.visits.append(last_visit)


class Visit:
    def __init__(self, camera):
        self.camera = camera
        self.what = None
        self.tracks = []
        self.start_time = None
        self.end_time = None
        self.score = None

    def add_track(self, track, lbl, score):
        if self.start_time is None:
            self.start_time = track.start_time
        self.tracks.append(track)
        self.end_time = track.start_time + timedelta(seconds=track.duration)

        if self.score is None or score > self.score:
            self.what = lbl


def process_job(queue, dataset, model_file, train_config, results_queue):

    classifier = KerasModel(train_config=train_config)
    classifier.load_weights(model_file)
    logging.info("Loaded model")
    i = 0
    results = {}
    while True:
        i += 1
        track = queue.get()
        print("processing", track)
        try:
            if track == "DONE":
                break
            else:
                tag = track.label
                if not tag:
                    continue
                tag = dataset.mapped_label(tag)
                track_data = dataset.db.get_track(track.clip_id, track.track_id)
                track_prediction = classifier.classify_track(
                    track.track_id, track_data, regions=track.track_bounds
                )
                mean = np.mean(track_prediction.original, axis=0)
                max_lbl = np.argmax(mean)
                predicted_lbl = classifier.labels[max_lbl]
                cam_results = results.setdefault(
                    track.camera, CameraResults(track.camera)
                )
                cam_results.add_result(track, predicted_lbl, np.amax(mean))

                # result = results.setdefault(
                #     tag,
                #     {
                #         "correct": 0,
                #         "correct_acc": [],
                #         "correct_ids": [],
                #         "incorrect": [],
                #         "total": 0,
                #         "incorrect_ids": [],
                #         "incorrect_acc": [],
                #     },
                # )
                # result["total"] += 1
                # if predicted_lbl == tag:
                #     result["correct"] += 1
                #     # stat["correct_acc"].append(track_prediction.score())
                #     result["correct_ids"].append(track.unique_id)
                #
                # else:
                #     result["incorrect_ids"].append(track.unique_id)
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s", e)
    # results_queue.put(results)
    results_queue.put(results)
    print("adding ", results)
    return


class ModelEvalute:
    def __init__(self, config, model_file, type):
        self.model_file = model_file
        self.classifier = None
        self.config = config
        # self.load_classifier(model_file, type)
        # self.db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))

    def load_classifier(self, model_file, type):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        t0 = datetime.now()
        logging.info("classifier loading %s", model_file)

        self.classifier = KerasModel(train_config=self.config.train, type=type)
        self.classifier.load_weights(model_file)

        logging.info("classifier loaded ({})".format(datetime.now() - t0))

    def save_confusion(self, dataset_file, output_file):

        datasets = pickle.load(open(dataset_file, "rb"))
        dataset = datasets[1]
        dataset.binarize(
            ["wallaby"],
            lbl_one="wallaby",
            lbl_two="not",
            keep_fp=False,
            scale=False,
            shuffle=False,
        )
        for label in dataset.labels:
            print(label, dataset.get_counts(label))
        self.classifier.confusion(dataset, output_file)

    def evaluate_dataset(self, dataset_file, tracks=False):
        datasets = pickle.load(open(dataset_file, "rb"))
        dataset = datasets[2]
        dataset.binarize(
            ["wallaby"],
            lbl_one="wallaby",
            lbl_two="not",
            keep_fp=False,
            scale=False,
            shuffle=False,
        )
        for label in dataset.labels:
            print(label, dataset.get_counts(label))
        if tracks:
            self.evaluate_tracks(dataset)
            return

        print()
        results = self.classifier.evaluate(dataset)
        print("Dataset", dataset_file, "loss,acc", results)

    def evaluate_tracks(self, dataset):
        dataset.set_read_only(True)
        results_queue = Queue()

        job_queue = Queue()
        processes = []
        for i in range(max(1, config.worker_threads)):
            p = Process(
                target=process_job,
                args=(
                    job_queue,
                    dataset,
                    self.model_file,
                    self.config.train,
                    results_queue,
                ),
            )
            processes.append(p)
            p.start()
        for track in dataset.tracks[:2]:
            if track.label and track.camera == "Hubble 4-2":
                job_queue.put(track)
        for i in range(len(processes)):
            job_queue.put("DONE")
        # for process in processes:
        #     print("join", process)
        #     process.join()
        # print("joined", results_queue.qsize())
        stats = {}
        merged_results = {}
        for i in range(len(processes)):
            result = results_queue.get()
            print("merge results", result)
            for key, cam_result in result.items():
                if key in merged_results:
                    merged_results[key].track_results.extend(cam_result.track_results)
                else:
                    merged_results[key] = cam_result
        for cam in merged_results.values():
            cam.calc_visits()
            print(cam.camera)
            for visit in cam.visits:
                print(visit.start_time, visit.end_time, visit.what, visit.tracks)
        # while not results_queue.empty():
        #     result = results_queue.get()
        #     print("merge results", result)
        #     for key, value in result.items():
        #         if key in stats:
        #             stat = stats[key]
        #             for r_key, r_value in value.items():
        #                 if isinstance(r_value, list):
        #                     stat.setdefault(r_key, []).extend(r_value)
        #                 else:
        #                     existing = stat.get(r_key, 0)
        #                     stat[r_key] = r_value + existing
        #         else:
        #             stats[key] = value
        #
        # print(stats)
        # for k, v in stats.items():
        #     correct_per = round(100 * float(v["correct"]) / v["total"])
        #     print("Stats for {} {}% correct".format(k, correct_per))
        #
        # # break

    def save_track(self, clip_id, track_id, type=5):
        track_data = self.db.get_track(clip_id, track_id)
        track_meta = self.db.get_track_meta(clip_id, track_id)
        seg_data = track_data[0:25]
        median = []
        for f in seg_data:
            median.append(np.median(f[0]))
        data = Preprocessor.apply(seg_data, median, default_inset=0,)

        preprocess_movement(
            track_data, data, 5, track_meta["bounds_history"], 1, type=type
        )


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )
    parser.add_argument("-t", "--dataset", help="Dataset file to use")
    parser.add_argument("--confusion", help="Save confusion matrix image")
    parser.add_argument(
        "--tracks", action="count", help="Evaluate whole track rather than samples"
    )
    parser.add_argument("--type", type=int, help="training type")

    parser.add_argument("-d", "--date", help="Use clips after this")
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
model_file = config.classify.model
if args.model_file:
    model_file = args.model_file
ev = ModelEvalute(config, model_file, args.type)
date = None
if args.date:
    date = parse(args.date)

# ev.save_track("645661", "269585")

if args.dataset:
    dataset_file = args.dataset
else:
    dataset_file = dataset_db_path(config)
if args.confusion is not None:
    ev.save_confusion(dataset_file, args.confusion)
else:
    ev.evaluate_dataset(dataset_file, args.tracks)
