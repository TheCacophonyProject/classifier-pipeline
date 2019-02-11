"""
Script to classify animals within a CPTV video file.
"""


import argparse
import json
import os
import logging
import sys
import datetime

import numpy as np

from classify.trackprediction import TrackPrediction
from classify.clipclassifier import ClipClassifier
import classify.globals as globs
from ml_tools import tools
from ml_tools.config import Config
from track.trackextractor import TrackExtractor
from track.track import Track


HERE = os.path.dirname(__file__)
RESOURCES_PATH = os.path.join(HERE, "resources")

# folders that are not processed when run with 'all'
IGNORE_FOLDERS = ['untagged', 'cat', 'dog', 'insect', 'unidentified', 'rabbit', 'hard', 'multi', 'moving', 'mouse',
                  'bird-kiwi', 'for_grant']


def resource_path(name):
    return os.path.join(RESOURCES_PATH, name)


def log_to_stdout():
    """ Outputs all log entries to standard out. """
    # taken from https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'source', help='a CPTV file to process, or a folder name, or "all" for all files within subdirectories of source folder.')

    parser.add_argument('-p', '--create-previews', action='count', help='Create MP4 previews for tracks (can be slow)')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')

    parser.add_argument('-q', '--high-quality-optical-flow', default=False,
                        action='store_true', help='Enabled higher quality optical flow (slow)')
    parser.add_argument('-f', '--force-overwrite', default='none',
                        help='Overwrite mode.  Options are all, old, or none.')

    parser.add_argument(
        '--start-date', help='Only clips on or after this day will be processed (format YYYY-MM-DD)')
    parser.add_argument(
        '--end-date', help='Only clips on or before this day will be processed (format YYYY-MM-DD)')
    # parser.add_argument('-b', '--side-by-side', default=False, action='store_true',
    #                     help='Output processed footage next to original output in preview MPEG')
    # parser.add_argument(
    #     '-m', '--model', help='Model to use for classification')
    # parser.add_argument('-i', '--include-prediction-in-filename', default=False,
    #                     action='store_true', help='Adds class scores to output files')

    conf = Config.read_default_config_file()

    args = parser.parse_args()

    if args.create_previews:
        conf["classify"]["preview"] = True

    # override verbose if true
    if args.verbose:
        conf["classify-tracking"]["verbose"] = True

    # if not args.model:
    #     print("setting model")
    #     conf["classify"]["model"]= args.model

    config = Config.load_from_map(conf)

    clip_classifier = ClipClassifier(config, config.classify_tracking)
    # clip_classifier.include_prediction_in_filename = args.include_prediction_in_filename

    if not config.classify.meta_to_stdout:
        log_to_stdout()

    if config.classify.preview:
        logging.info("Creating previews")

    if not config.use_gpu:
        logging.info("GPU mode disabled.")

    if not os.path.exists(config.classify.model + ".meta"):
        logging.error("No model found named '{}'.".format(config.classify.model+".meta"))
        exit(13)

    if args.start_date:
        clip_classifier.start_date = datetime.strptime(
            args.start_date, "%Y-%m-%d")

    if args.end_date:
        clip_classifier.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # just fetch the classifier now so it doesn't impact the benchmarking on the first clip analysed.
    _ = clip_classifier.classifier

    # set overwrite mode
    if args.force_overwrite.lower() not in ['all', 'old', 'none']:
        raise Exception("Valid overwrite modes are all, old, or none.")
    clip_classifier.overwrite_mode = args.force_overwrite.lower()

    # set verbose
    clip_classifier.verbose = args.verbose if args.verbose is not None else 0

    if args.source == "all":
        clip_classifier.process_all(args.source_folder)
    elif os.path.splitext(args.source)[-1].lower() == '.cptv':
        source_file = tools.find_file_from_cmd_line(config.source_folder, args.source)
        if source_file is None:
            return
        logging.info("Processing file '" + source_file + "'")
        clip_classifier.process_file(source_file)
    else:
        clip_classifier.process_folder(
            os.path.join(config.source_folder, args.source))


if __name__ == "__main__":
    main()
