import argparse
import os
import logging
from datetime import datetime

from .clipclassifier import ClipClassifier
from ml_tools.logs import init_logging
from ml_tools import tools
from config.config import Config
from ml_tools.previewer import Previewer
import absl.logging


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
        "--create-previews",
        action="count",
        help="Create MP4 previews for tracks (can be slow)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", help="Display additional information."
    )
    parser.add_argument(
        "--start-date",
        help="Only clips on or after this day will be processed (format YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="Only clips on or before this day will be processed (format YYYY-MM-DD)",
    )
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--processor-folder",
        help="When running from thermal-processing use this to specify the folder for both the source cptv and output mp4. With this option the metadata will be sent to stdout.",
    )
    parser.add_argument(
        "-T", "--timestamps", action="store_true", help="Emit log timestamps"
    )
    args = parser.parse_args()

    config = Config.load_from_file(args.config_file)
    config.validate()
    init_logging(args.timestamps)

    # parse command line arguments
    if args.create_previews:
        config.classify.preview = Previewer.PREVIEW_CLASSIFIED

    if args.verbose:
        config.classify_tracking.verbose = True

    if args.processor_folder:
        config.classify.meta_to_stdout = True
        config.base_data_folder = args.processor_folder
        config.classify.classify_folder = args.processor_folder
        config.source_folder = args.processor_folder

    clip_classifier = ClipClassifier(config, config.classify_tracking)

    # parse start and end dates
    if args.start_date:
        clip_classifier.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        clip_classifier.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if config.classify.preview != Previewer.PREVIEW_NONE:
        logging.info("Creating previews")

    if not config.use_gpu:
        logging.info("GPU mode disabled.")

    if not os.path.exists(config.classify.model + ".meta"):
        logging.error(
            "No model found named '{}'.".format(config.classify.model + ".meta")
        )
        exit(13)

    # just fetch the classifier now so it doesn't impact the benchmarking on the first clip analysed.
    _ = clip_classifier.classifier

    if os.path.splitext(args.source)[-1].lower() == ".cptv":
        source_file = tools.find_file_from_cmd_line(config.source_folder, args.source)
        if source_file is None:
            return
        clip_classifier.process_file(source_file)
    else:
        folder = config.source_folder
        if args.source != "all":
            os.path.join(config.source_folder, folder)
        clip_classifier.process_all(folder)


if __name__ == "__main__":
    main()
