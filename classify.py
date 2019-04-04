"""
Script to classify animals within a CPTV video file.
"""


import argparse
import os
import logging
import sys
from datetime import datetime

from classify.clipclassifier import ClipClassifier
from ml_tools import tools
from ml_tools.config import Config
from ml_tools.previewer import Previewer

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
    parser.add_argument(
        '--start-date', help='Only clips on or after this day will be processed (format YYYY-MM-DD)')
    parser.add_argument(
        '--end-date', help='Only clips on or before this day will be processed (format YYYY-MM-DD)')
    parser.add_argument('-c', '--config-file', help="Path to config file to use")
    parser.add_argument('--processor-folder', help="When running from thermal-processing use this to specify the folder for both the source cptv and output mp4. With this option the metadata will be sent to stdout.")

    args = parser.parse_args()
    config = Config.load_from_file(args.config_file)

    # parse command line arguments
    if args.create_previews:
        config.classify.preview = Previewer.PREVIEW_CLASSIFIED

    if args.verbose:
        config.classify_tracking.verbose = True

    if args.processor_folder:
        config.classify.meta_to_stdout = True
        config.base_data_folder = args.processor_folder
        config.classify.classify_folder = ''
        config.source_folder = ''

    clip_classifier = ClipClassifier(config, config.classify_tracking)

    # parse start and end dates
    if args.start_date:
        clip_classifier.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        clip_classifier.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if not config.classify.meta_to_stdout:
        log_to_stdout()

    if config.classify.preview != Previewer.PREVIEW_NONE:
        logging.info("Creating previews")

    if not config.use_gpu:
        logging.info("GPU mode disabled.")

    if not os.path.exists(config.classify.model + ".meta"):
        logging.error("No model found named '{}'.".format(config.classify.model+".meta"))
        exit(13)

    # just fetch the classifier now so it doesn't impact the benchmarking on the first clip analysed.
    _ = clip_classifier.classifier

    if args.source == "all":
        clip_classifier.process_all(config.source_folder)
    elif os.path.splitext(args.source)[-1].lower() == '.cptv':
        source_file = tools.find_file_from_cmd_line(config.source_folder, args.source)
        if source_file is None:
            return
        clip_classifier.process_file(source_file)
    else:
        clip_classifier.process_folder(
            os.path.join(config.source_folder, args.source))

if __name__ == "__main__":
    main()
