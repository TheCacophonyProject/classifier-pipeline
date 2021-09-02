import argparse
import os
import logging
from datetime import datetime

from track.trackextractor import TrackExtractor
from ml_tools.logs import init_logging
from config.config import Config
from config.classifyconfig import ModelConfig

import absl.logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "source",
        help='a CPTV file to process, or a folder name, or "all" for all files within subdirectories of source folder.',
    )
    parser.add_argument(
        "-p",
        "--preview-type",
        help="Create MP4 previews of this type (can be slow), this overrides the config",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", help="Display additional information."
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--processor-folder",
        help="When running from thermal-processing use this to specify the folder for both the source cptv and output mp4. With this option the metadata will be sent to stdout.",
    )
    parser.add_argument(
        "-T", "--timestamps", action="store_true", help="Emit log timestamps"
    )

    parser.add_argument(
        "--cache",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Dont keep video frames in memory for classification later, but cache them to disk (Best for large videos, but slower)",
    )
    args = parser.parse_args()
    config = Config.load_from_file(args.config_file)
    config.validate()
    init_logging(args.timestamps)

    # parse command line arguments
    if args.preview_type:
        config.classify.preview = args.preview_type

    if args.verbose:
        config.classify_tracking.verbose = True

    if args.processor_folder:
        config.classify.meta_to_stdout = True
        config.base_data_folder = args.processor_folder
        config.classify.classify_folder = args.processor_folder
        config.source_folder = args.processor_folder

    extractor = TrackExtractor(config, cache_to_disk=args.cache)

    extractor.extract(os.path.join(args.source))


if __name__ == "__main__":
    main()
