import attr
import absl.logging
import argparse
import os
import logging
import math
import sys
from enum import Enum
from pathlib import Path
from cacophonyapi.user import UserAPI

from ml_tools import tools
from config.config import Config

from track.trackextractor import TrackExtractor
from .testconfig import TestConfig

MATCH_ERROR = 1


@attr.s
class Summary:
    better_tracking = attr.ib(default=0)
    same_tracking = attr.ib(default=0)
    worse_tracking = attr.ib(default=0)
    classify_incorrect = attr.ib(default=0)
    classified_correct = attr.ib(default=0)
    total_tests = attr.ib(default=0)
    unmatched_tests = attr.ib(default=0)
    unmatched_tracks = attr.ib(default=0)

    def update(self, other):
        self.better_tracking += other.better_tracking
        self.same_tracking += other.same_tracking
        self.worse_tracking += other.worse_tracking
        self.classify_incorrect += other.classify_incorrect
        self.classified_correct += other.classified_correct
        self.total_tests += other.total_tests
        self.unmatched_tests += other.unmatched_tests
        self.unmatched_tracks += other.unmatched_tracks

    @property
    def classified_percentage(self):
        if self.total_tests == 0:
            return 0
        return round(100.0 * self.classified_correct / self.total_tests)

    @property
    def tracked_well_percentage(self):
        if self.total_tests == 0:
            return 0
        return round(
            100.0 * (self.same_tracking + self.better_tracking) / self.total_tests
        )

    def print_summary(self):
        print("===== OVERAL =====")
        print(
            "Classify Results {}% {}/{}".format(
                self.classified_percentage,
                self.classified_correct,
                self.total_tests,
            )
        )
        print(
            "Tracking Results Better/Same {}% {}/{} With {} unmatched tracks (false-positives) and {} missed tests".format(
                self.tracked_well_percentage,
                self.same_tracking + self.better_tracking,
                self.total_tests,
                self.unmatched_tracks,
                self.unmatched_tests,
            )
        )


class TrackingStatus(Enum):
    IMPROVED = 1
    SAME = 0
    WORSE = -1


def match_track(gen_track, expected_tracks):
    score = None
    match = None
    MAX_ERROR = 8
    for track in expected_tracks:
        start_diff = abs(track.start - gen_track.start_s)

        gen_start = gen_track.bounds_history[0]
        distance = tools.eucl_distance_sq(
            (track.start_pos.mid_x, track.start_pos.mid_y),
            (gen_start.mid_x, gen_start.mid_y),
        )
        distance += tools.eucl_distance_sq(
            (track.start_pos.x, track.start_pos.y), (gen_start.x, gen_start.y)
        )
        distance += tools.eucl_distance_sq(
            (track.start_pos.right, track.start_pos.bottom),
            (gen_start.right, gen_start.bottom),
        )
        distance /= 3.0
        distance = math.sqrt(distance)

        # makes it more comparable to start error
        distance /= 4.0
        new_score = distance + start_diff
        if new_score > MAX_ERROR:
            continue
        if score is None or new_score < score:
            match = track
            score = new_score
    return match


class RecordingMatch:
    def __init__(self, filename, id_):
        self.matches = []
        self.unmatched_tracks = []
        self.unmatched_tests = []
        self.filename = filename
        self.id = id_
        self.summary = Summary()

    def match(self, test, tracks, predictions=None):
        self.summary.total_tests += len(test.tracks)
        gen_tracks = sorted(tracks, key=lambda x: x.get_id())
        gen_tracks = sorted(gen_tracks, key=lambda x: x.start_s)

        self.unmatched_tests = set(test.tracks)
        predicted_tag = None
        for i, track in enumerate(gen_tracks):
            if predictions is not None:
                prediction = predictions.prediction_for(track.get_id())
                predicted_tag = prediction.predicted_tag()
            test_track = match_track(track, self.unmatched_tests)
            if test_track is not None:
                self.unmatched_tests.remove(test_track)
                match = Match(test_track, track, predicted_tag)
                self.new_match(match)
            else:
                self.unmatched_tracks.append((predicted_tag, track))
                self.summary.unmatched_tracks += 1
                print(
                    "Unmatched track start {} end {}".format(
                        track.start_s,
                        track.end_s,
                    )
                )
        self.summary.unmatched_tests = len(self.unmatched_tests)

    def new_match(self, match):
        if match.status == TrackingStatus.IMPROVED:
            self.summary.better_tracking += 1
        elif match.status == TrackingStatus.SAME:
            self.summary.same_tracking += 1
        else:
            self.summary.worse_tracking += 1
        if match.tag_match():
            self.summary.classified_correct += 1
        else:
            self.summary.classify_incorrect += 1
        self.matches.append(match)

    def print_summary(self):
        print("*******{} Classification Results ******".format(self.id))
        print(
            "matches {}\tmismatches {}\tunmatched {}".format(
                self.summary.classified_correct,
                self.summary.classify_incorrect,
                self.summary.unmatched_tracks,
            )
        )
        print("*******Tracking******")
        print(
            "same {} better {}\t worse {}".format(
                self.summary.same_tracking,
                self.summary.better_tracking,
                self.summary.worse_tracking,
            )
        )
        if self.summary.unmatched_tests > 0:
            print("unmatched tests {}\t ".format(self.summary.unmatched_tests))

    def write_results(self, f):
        f.write("{}{}{}\n".format("-" * 10, "Recording", "-" * 90))
        f.write("Recordings[{}] {}\n\n".format(self.id, self.filename))
        for match in self.matches:
            match.write_results(f)

        if self.summary.unmatched_tracks > 0:
            f.write("Unmatched Tracks\n")
        for what, track in self.unmatched_tracks:
            f.write(
                "{} - [{}s]Start-End {} - {}\n".format(
                    what,
                    round(track.end_s - track.start_s, 1),
                    round(track.start_s, 1),
                    round(track.end_s, 1),
                )
            )
        f.write("\n")

        if self.summary.unmatched_tests > 0:
            f.write("Unmatched Tests\n")
        for expected in self.unmatched_tests:
            f.write(
                "{} - Opt[{}s] Start-End {} - {}, Expected[{}s] {} - {}\n".format(
                    expected.tag,
                    round(expected.opt_end - expected.opt_start, 1),
                    expected.opt_start,
                    expected.opt_end,
                    round(expected.end - expected.start, 1),
                    expected.start,
                    expected.end,
                )
            )
        f.write("\n")


class Match:
    def __init__(self, test_track, track, tag=None):
        expected_length = test_track.opt_end - test_track.opt_start
        self.length_diff = round(expected_length - (track.end_s - track.start_s), 2)
        self.start_diff_s = round(test_track.start - track.start_s, 2)
        self.end_diff_s = round(test_track.end - track.end_s, 2)
        self.opt_start_diff_s = round(test_track.opt_start - track.start_s, 2)
        self.opt_end_diff_s = round(test_track.opt_end - track.end_s, 2)
        self.error = round(abs(self.opt_start_diff_s) + abs(self.opt_end_diff_s), 1)

        if self.error <= test_track.calc_error():
            self.status = TrackingStatus.IMPROVED
        elif self.error < MATCH_ERROR:
            self.status = TrackingStatus.SAME
        else:
            self.status = TrackingStatus.WORSE
        self.expected_tag = test_track.tag
        self.got_animal = tag
        self.test_track = test_track
        self.track = track

    def tracking_status(self):
        if self.status == TrackingStatus.IMPROVED:
            return "Better Tracking"
        elif self.status == TrackingStatus.SAME:
            return "Same Tracking"
        return "Worse Tracking"

    def classify_status(self):
        if self.expected_tag == self.got_animal:
            return "Classified Correctly"
        return "Classified Incorrect"

    def write_results(self, f):
        f.write("{}{}{}\n".format("=" * 10, "Track", "=" * 90))

        f.write("{}\t{}\n".format(self.tracking_status(), self.classify_status()))
        f.write("Exepcted:\n")
        f.write(
            "{} - Opt[{}s] Start-End {} - {}, Expected[{}s] {} - {}\n".format(
                self.expected_tag,
                self.test_track.opt_length(),
                self.test_track.opt_start,
                self.test_track.opt_end,
                self.test_track.length(),
                self.test_track.start,
                self.test_track.end,
            )
        )
        f.write("Got:\n")
        f.write(
            "{} - [{}s]Start-End {} - {}\n".format(
                self.got_animal,
                round(self.track.end_s - self.track.start_s, 1),
                round(self.track.start_s, 1),
                round(self.track.end_s, 1),
            )
        )
        f.write("\n")

    def tag_match(self):
        return self.expected_tag == self.got_animal


class TestClassify:
    def __init__(self, args):
        self.test_config = TestConfig.load_from_file(args.tests)
        self.classifier_config = Config.load_from_file(args.classify_config)
        if args.model_file:
            model_file = args.model_file
        self.track_extractor = TrackExtractor(self.classifier_config)
        # try download missing tests
        if args.user and args.password:
            api = UserAPI(args.server, args.user, args.password)
            out_dir = Path(self.test_config.clip_dir)
            if not out_dir.exists():
                out_dir.mkdir()
            for test in self.test_config.recording_tests:
                filepath = out_dir / test.filename
                if not filepath.exists():
                    if iter_to_file(
                        out_dir / test.filename, api.download_raw(test.rec_id)
                    ):
                        logging.info("Saved %s", filepath)
        self.results = []

    def run_tests(self, args):
        out_dir = Path(self.test_config.clip_dir)

        for test in self.test_config.recording_tests:
            filepath = out_dir / test.filename
            if not filepath.exists():
                logging.info(
                    " %s not found, add cmd args user and password to download this file from the cacophony server",
                    filepath,
                )
                continue
            logging.info("testing %s ", test.filename)
            clip = self.track_extractor.extract_tracks(filepath)
            meta_filename = filepath.with_suffix(".txt")
            self.track_extractor.save_metadata(str(filepath), meta_filename, clip, 0)

            predictions = None
            rec_match = self.compare_output(clip, predictions, test)
            if self.track_extractor.previewer:
                mpeg_filename = filepath.with_suffix(".mp4")
                logging.info("Exporting preview to '%s'", mpeg_filename)
                self.track_extractor.previewer.export_clip_preview(
                    mpeg_filename, clip, predictions
                )
            self.results.append(rec_match)

    def write_results(self, filename, config_file):
        with open(filename, "w") as f:
            for res in self.results:
                res.write_results(f)
            f.write("Config\n")
            with open(config_file, "r") as config:
                f.write(config.read())

    def print_summary(self):
        print("===== SUMMARY =====")
        total_summary = Summary()
        for result in self.results:
            result.print_summary()
            total_summary.update(result.summary)

        total_summary.print_summary()

    def compare_output(self, clip, predictions, expected):
        rec_match = RecordingMatch(clip.source_file, expected.rec_id)
        rec_match.match(expected, clip.tracks, predictions)
        return rec_match


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--classify_config",
        default="./classifier.yaml",
        help="config file to classify with",
    )

    parser.add_argument(
        "-t",
        "--tests",
        default="tests/tracking-tests.yml",
        help="YML file containing tests",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="tests/tracking-results.txt",
        help="Text file descriibing the results",
    )

    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )

    parser.add_argument("--user", help="API server username")
    parser.add_argument("--password", help="API server password")
    parser.add_argument(
        "-s",
        "--server",
        default="https://api.cacophony.org.nz",
        help="CPTV file server URL",
    )
    args = parser.parse_args()
    return args


def init_logging():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def main():
    init_logging()
    args = parse_args()
    test = TestClassify(args)
    test.run_tests(args)
    test.write_results(args.output, args.classify_config)
    test.print_summary()


if __name__ == "__main__":
    main()


def iter_to_file(filename, source, overwrite=True):
    if not overwrite and Path(filename).is_file():
        print("{} already exists".format(filename))
        return False
    with open(filename, "wb") as f:
        for chunk in source:
            f.write(chunk)
    return True
