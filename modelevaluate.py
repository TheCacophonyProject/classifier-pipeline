import argparse
import logging
import pickle
import time
import sys

from dateutil.parser import parse
from config.config import Config
from datetime import datetime, timedelta
import numpy as np

from multiprocessing import Process, Queue
from ml_tools.datagenerator import preprocess_movement
from ml_tools.dataset import Preprocessor
from ml_tools.framedataset import dataset_db_path
from ml_tools.kerasmodel import KerasModel

VISIT_INTERVAL = 10 * 60


class SmallTrack:
    def __init__(self, track):
        self.camera = track.camera
        self.clip_id = track.clip_id
        self.track_id = track.track_id
        self.label = track.label
        # date and time of the start of the track
        self.start_time = track.start_time
        self.start_frame = track.start_frame
        # duration in seconds
        self.duration = track.duration

    def __repr__(self):
        return "{}-{}".format(self.clip_id, self.track_id)


class CameraResults:
    def __init__(self, camera):
        self.camera = camera
        self.track_results = []
        self.visits = []

    def add_result(self, track, predicted_lbl, prediction_score, sure, expected):
        self.track_results.append(
            (track, predicted_lbl, prediction_score, sure, expected)
        )

    def calc_visits(self):
        start_sorted = sorted(self.track_results, key=lambda res: res[0].start_time)
        last_visit = None
        self.visits = []
        for result in start_sorted:
            if last_visit is None:
                last_visit = Visit(self.camera, result[4])
                last_visit.add_track(*result)
                self.visits.append(last_visit)
            else:
                time_diff = result[0].start_time - last_visit.end_time
                if time_diff.total_seconds() < VISIT_INTERVAL:
                    same_lbl = result[1] == last_visit.what

                    if same_lbl or not result[3]:
                        last_visit.add_track(*result)
                        continue
                last_visit = Visit(self.camera, result[4])
                last_visit.add_track(*result)
                self.visits.append(last_visit)


class Visit:
    def __init__(self, camera, expected):
        self.camera = camera
        self.what = None
        self.tracks = []
        self.start_time = None
        self.end_time = None
        self.score = None
        self.expected = expected
        self.sure = None

    def add_track(self, track, lbl, score, sure, expected):
        if self.start_time is None:
            self.sure = sure
            self.start_time = track.start_time

        self.tracks.append(track)
        self.end_time = track.start_time + timedelta(seconds=track.duration)
        if self.what is None or not self.sure and sure:
            self.what = lbl
            self.sure = self.sure or sure


def process_job(queue, dataset, model_file, train_config, results_queue):

    classifier = KerasModel(train_config=train_config)
    classifier.load_weights(model_file)
    logging.info("Loaded model")
    i = 0

    while True:
        i += 1
        track = queue.get()
        try:
            if track == "DONE":
                break
            else:
                expected_tag = track.label
                if not expected_tag:
                    continue
                expected_tag = dataset.mapped_label(expected_tag)
                track_data = dataset.db.get_track(track.clip_id, track.track_id)
                track_prediction = classifier.classify_track(
                    track.track_id, track_data, regions=track.track_bounds
                )
                counts = [0, 0]
                for pred in track_prediction.original:
                    counts[np.argmax(pred)] += 1
                mean = np.mean(track_prediction.original, axis=0)
                max_lbl = np.argmax(mean)
                predicted_lbl = classifier.labels[max_lbl]
                vel_sum = [abs(vel[0]) + abs(vel[1]) for vel in track.frame_velocity]

                sure = True
                if expected_tag == "wallaby":
                    total = counts[0] + counts[1]
                    wallaby_tagged = counts[0] / total > 0.1
                    max_perc = np.amax(mean)

                    sure = max_perc > 0.85 and not wallaby_tagged
                    # require movement to be sure of not
                    sure = sure and np.mean(vel_sum) > 0.6

                pickled_track = pickle.dumps(SmallTrack(track))
                results_queue.put(
                    (pickled_track, predicted_lbl, np.amax(mean), sure, expected_tag)
                )
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s", e)
    return


class ModelEvalute:
    def __init__(self, config, model_file):
        self.model_file = model_file
        self.classifier = None
        self.config = config
        # self.load_classifier(model_file, type)
        # self.db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))

    def load_classifier(self):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        t0 = datetime.now()
        logging.info("classifier loading %s", self.model_file)

        self.classifier = KerasModel(train_config=self.config.train)
        self.classifier.load_weights(self.model_file)

        logging.info("classifier loaded ({})".format(datetime.now() - t0))

    def save_confusion(self, dataset, output_file):
        self.load_classifier()
        self.classifier.confusion(dataset, output_file)

    def evaluate_dataset(self, dataset, tracks=False):
        for label in dataset.labels:
            print(label, dataset.get_counts(label))
        if tracks:
            self.evaluate_tracks(dataset)
            return

        self.load_classifier()
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
        for track in dataset.tracks:
            job_queue.put(track)
        for i in range(len(processes)):
            job_queue.put("DONE")

        cam_results = {}
        overall_stats = {}
        while True:
            alive = [p for p in processes if p.is_alive()]
            if results_queue.empty():
                if len(alive) == 0:
                    logging.info("got all results")
                    break
                else:
                    time.sleep(2)
                    continue
            result = results_queue.get()

            track = pickle.loads(result[0])
            predicted_lbl = result[1]
            mean = result[2]
            sure = result[3]
            expected_lbl = result[4]

            cam_result = cam_results.setdefault(
                track.camera, CameraResults(track.camera)
            )
            cam_result.add_result(
                track, predicted_lbl, np.amax(mean), sure, expected_lbl
            )

            result = overall_stats.setdefault(
                expected_lbl,
                {
                    "correct": 0,
                    "correct_acc": [],
                    "correct_ids": [],
                    "incorrect": 0,
                    "total": 0,
                    "incorrect_ids": [],
                    "incorrect_acc": [],
                },
            )
            result["total"] += 1
            if predicted_lbl == expected_lbl:
                result["correct"] += 1
            else:
                result["incorrect"] += 1

        # visit stats
        visit_stats = {}
        for cam in cam_results.values():
            cam.calc_visits()
            print("Camera", cam.camera)
            for visit in cam.visits:
                v_stat = visit_stats.setdefault(
                    visit.expected, {"correct": 0, "incorrect": 0}
                )
                if visit.expected == visit.what:
                    v_stat["correct"] += len(visit.tracks)
                else:
                    v_stat["incorrect"] += len(visit.tracks)
                print(
                    "Visit Start {} - End {} What {} Expected {}, Tracks {}".format(
                        visit.start_time,
                        visit.end_time,
                        visit.what,
                        visit.expected,
                        visit.tracks,
                    )
                )
        print("visit _stats", visit_stats)
        for k, v in visit_stats.items():
            correct_per = round(
                100 * float(v["correct"]) / (v["correct"] + v["incorrect"])
            )
            print("Visit Stats for {} {}% correct".format(k, correct_per))

        #
        print("Non visit stats")
        for k, v in overall_stats.items():
            correct_per = round(100 * float(v["correct"]) / v["total"])
            print("Stats for {} {}% correct".format(k, correct_per))

        # # break

    def save_track(self, clip_id, track_id, db, type=5):
        track_data = db.get_track(clip_id, track_id)
        track_meta = db.get_track_meta(clip_id, track_id)
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
    parser.add_argument(
        "-s",
        "--dataset",
        type=int,
        default=2,
        help="Dataset to use 0 - Training, 1 - Validation, 2 - Train (Default)",
    )
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
ev = ModelEvalute(config, model_file)
date = None
if args.date:
    date = parse(args.date)


dataset_file = dataset_db_path(config)


datasets = pickle.load(open(dataset_file, "rb"))
dataset = datasets[args.dataset]
ev.save_track("575914", "249216", dataset.db)
dataset.binarize(
    ["wallaby"],
    lbl_one="wallaby",
    lbl_two="not",
    keep_fp=False,
    scale=False,
    shuffle=False,
)
logging.info(
    "Dataset loaded %s, using labels %s, mapped labels %s",
    dataset.name,
    dataset.labels,
    dataset.label_mapping,
)
logging.info("%s %s / %s / %s", "label", "segments", "frames", "tracks")
for label in dataset.labels:
    segments, frames, tracks, _, _ = dataset.get_counts(label)
    logging.info("%s %s / %s / %s", label, segments, frames, tracks)

if args.confusion is not None:
    ev.save_confusion(dataset, args.confusion)
else:
    ev.evaluate_dataset(dataset, args.tracks)
