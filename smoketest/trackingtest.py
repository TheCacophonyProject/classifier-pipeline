import absl.logging
import argparse
import os
import json
import logging
import sys
from pathlib import Path
from ml_tools import tools
from config.config import Config
from classify.clipclassifier import ClipClassifier
from .testconfig import TestConfig


class RecordingMatch:
    def __init__(self, filename, id_):
        self.matches = []
        self.unmatched_tracks = []
        self.unmatched_tests = []
        self.filename = filename
        self.id = id_

    def match(self, expected, tracks, predictions):
        expected_tracks = sorted(expected.tracks, key=lambda x: x.start)
        expected_tracks = [track for track in expected_tracks if track.expected]

        gen_tracks = sorted(tracks, key=lambda x: x.get_id())
        gen_tracks = sorted(gen_tracks, key=lambda x: x.start_s)

        for i, track in enumerate(gen_tracks):
            prediction = predictions.prediction_for(track.get_id())
            if len(expected_tracks) > i:
                # just matching in order of start time, this could faile with multiple tracks
                # starting at same time
                expected = expected_tracks[i]
                match = Match(
                    expected, track, prediction.predicted_tag(predictions.labels)
                )
                self.matches.append(match)
            else:
                self.unmatched_tracks.append(
                    (prediction.predicted_tag(predictions.labels), track)
                )
                print(
                    "Unmatched track tag {} start {} end {}".format(
                        prediction.predicted_tag(predictions.labels),
                        track.start_s,
                        track.end_s,
                    )
                )
        if len(gen_tracks) < len(expected_tracks):
            self.unmatched_tests = expected_tracks[len(self.matches) :]

    def print_summary(self):
        print(self.filename)
        matched = [match for match in self.matches if match.tag_match()]
        unmatched = [match for match in self.matches if not match.tag_match()]
        better = [match for match in self.matches if match.improvement]
        worse = [match for match in self.matches if not match.improvement]
        print("*******Classifying******")
        print(
            "matches {}\tmismatches {}\tunmatched {}".format(
                len(matched), len(unmatched), len(self.unmatched_tracks)
            )
        )
        print("*******Tracking******")
        print("better {}\t worse {}".format(len(better), len(worse)))
        if len(self.unmatched_tests) > 0:
            print("unmatched tests {}\t ".format(len(self.unmatched_tests)))

    def write_results(self, f):
        f.write("{}{}{}\n".format("-" * 10, "Recording", "-" * 90))
        f.write("Recordings[{}] {}\n\n".format(self.id, self.filename))
        for match in self.matches:
            match.write_results(f)

        if len(self.unmatched_tracks) > 0:
            f.write("Unmatched Tracks\n")
        for (what, track) in self.unmatched_tracks:
            f.write(
                "{} - [{}s]Start-End {} - {}\n".format(
                    what,
                    round(track.end_s - track.start_s, 1),
                    round(track.start_s, 1),
                    round(track.end_s, 1),
                )
            )
        f.write("\n")

        if len(self.unmatched_tests) > 0:
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
    def __init__(self, expected, track, tag):
        expected_length = expected.opt_end - expected.opt_start
        self.length_diff = round(expected_length - (track.end_s - track.start_s), 2)
        self.start_diff_s = round(expected.start - track.start_s, 2)
        self.end_diff_s = round(expected.end - track.end_s, 2)
        self.opt_start_diff_s = round(expected.opt_start - track.start_s, 2)
        self.opt_end_diff_s = round(expected.opt_end - track.end_s, 2)
        self.error = round(abs(self.opt_start_diff_s) + abs(self.opt_end_diff_s), 1)
        self.improvement = self.error <= expected.calc_error()
        self.expected_tag = expected.tag
        self.got_animal = tag
        self.expected = expected
        self.track = track

    def write_results(self, f):
        f.write("{}{}{}\n".format("=" * 10, "Track", "=" * 90))
        tracking_s = "Improved Tracking" if self.improvement else "Worse Tracking"
        classify_s = (
            "Classified Correctly"
            if self.expected_tag == self.got_animal
            else "Classified Incorrect"
        )
        f.write("{}\t{}\n".format(tracking_s, classify_s))
        f.write("Exepcted:\n")
        f.write(
            "{} - Opt[{}s] Start-End {} - {}, Expected[{}s] {} - {}\n".format(
                self.expected_tag,
                round(self.expected.opt_end - self.expected.opt_start, 1),
                self.expected.opt_start,
                self.expected.opt_end,
                round(self.expected.end - self.expected.start, 1),
                self.expected.start,
                self.expected.end,
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

        self.classifier_config = Config.load_from_file(args.classify_config)
        model_file = self.classifier_config.classify.model
        if args.model_file:
            model_file = args.model_file

        path, ext = os.path.splitext(model_file)
        keras_model = False
        if ext == ".pb":
            keras_model = True
            print("is keras model")
        self.clip_classifier = ClipClassifier(
            self.classifier_config,
            self.classifier_config.classify_tracking,
            model_file,
            keras_model,
        )
        self.test_config = TestConfig.load_from_file(args.tests)
        self.results = []

    def run_tests(self):
        for test in self.test_config.recording_tests:
            logging.info("testing {} ".format(test.filename))
            clip, predictions = self.clip_classifier.classify_file(test.filename)
            rec_match = self.compare_output(clip, predictions, test)
            self.results.append(rec_match)

    def write_results(self):
        with open("smoketest-results.txt", "w") as f:
            for res in self.results:
                res.write_results(f)
            f.write("Config\n")
            json.dump(self.classifier_config, f, indent=2, default=convert_to_dict)

    def print_summary(self):
        print("===== SUMMARY =====")
        for result in self.results:
            result.print_summary()

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
        default="smoketest/tracking-tests.yml",
        help="YML file containing tests",
    )

    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
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
    test.run_tests()
    test.write_results()
    test.print_summary()


if __name__ == "__main__":
    main()


def convert_to_dict(obj):
    return obj.__dict__
