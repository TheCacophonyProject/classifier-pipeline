"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import argparse
import os

from ml_tools.logs import init_logging
from config.config import Config
from mldataset.makedataset import ClipLoader

from pathlib import Path


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-folder",
        default="clip_db",
        help="Folder to put processed files",
    )

    parser.add_argument(
        "target",
        default=None,
        help='Target to process, "all" processes all folders, "test" runs test cases, "clean" to remove banned clips from db, or a "cptv" file to run a single source.',
    )
    args = parser.parse_args()
    config = Config.load_from_file(None)
    args.out_folder = Path(args.out_folder)
    return config, args


def main():
    config, args = parse_params()
    init_logging()
    loader = ClipLoader(config)
    loader.process_all(args.target, args.out_folder)


if __name__ == "__main__":
    main()
