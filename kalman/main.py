from ml_tools.config import Config
from ml_tools import tools
from kalman.kalmanpredictor import KalmanPredictor
from track.cptvtrackextractor import CPTVTrackExtractor

import argparse
import logging
import os


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Target CPTV or MP3 file to process")
    args = parser.parse_args()
    config = Config.load_from_file(None)
    config.extract.preview = "none"

    if os.path.splitext(args.target)[1].lower() == ".cptv":
        # run single source
        source_file = tools.find_file_from_cmd_line(config.source_folder, args.target)

    return config, source_file


def create_tracker(config, source_file):
    # setup extractor
    extractor = CPTVTrackExtractor(config, config.tracking)
    logging.info("Processing file '" + source_file + "'")
    tag = os.path.basename(os.path.dirname(source_file))
    return extractor.process_file(source_file, tag=tag)


def main():
    config, source_file = parse_config()
    if source_file is None:
        return
    tracker = create_tracker(config, source_file)
    predictor = KalmanPredictor(config, source_file, tracker)
    # make kalman predictions
    predictor.make_preview()
