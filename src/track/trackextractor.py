from multiprocessing import Pool

import cv2
import gc
import json
import logging
import os.path
import time

import numpy as np

from classify.trackprediction import Predictions
from track.clip import Clip
from track.cliptrackextractor import ClipTrackExtractor
from track.irtrackextractor import IRTrackExtractor
from pathlib import Path
from ml_tools import tools
from ml_tools.previewer import Previewer
from track.track import Track

from classify.thumbnail import get_thumbnail_info, best_trackless_thumb


class TrackExtractor:
    """Generate tracks for CPTV files."""

    def __init__(self, config, cache_to_disk=None, retrack=False):
        """Create an instance of a clip classifier"""

        self.config = config
        self.worker_threads = max(1, config.worker_threads)
        self.retrack = retrack
        if cache_to_disk is None:
            self.cache_to_disk = self.config.classify.cache_to_disk
        else:
            self.cache_to_disk = cache_to_disk

    def get_meta_data(self, filename):
        """Reads meta-data for a given cptv file."""
        source_meta_filename = os.path.splitext(filename)[0] + ".txt"
        if os.path.exists(source_meta_filename):
            meta_data = tools.load_clip_metadata(source_meta_filename)

            tags = set()
            for record in meta_data["Tags"]:
                # skip automatic tags
                if record.get("automatic", False):
                    continue
                else:
                    tags.add(record["animal"])

            tags = list(tags)

            if len(tags) == 0:
                tag = "no tag"
            elif len(tags) == 1:
                tag = tags[0] if tags[0] else "none"
            else:
                tag = "multi"
            meta_data["primary_tag"] = tag
            return meta_data
        else:
            return None

    def extract(self, base, to_stdout=False):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        base = Path(base)
        if not base.exists():
            logging.error("Could not find file or directory %s", base)
            return
        if base.is_file():
            extract_file(base, self.config, self.cache_to_disk, self.retrack, to_stdout)
            return
        data = []
        for folder_path, _, files in os.walk(base):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".avi", ".cptv"]:
                    full_path = os.path.join(folder_path, name)
                    data.append(full_path)
        with Pool(
            self.worker_threads,
            init_worker,
            (self.config, self.cache_to_disk, to_stdout, self.retrack),
        ) as pool:
            pool.map(extract_thread, data)


config = None
cache_to_disk = None
to_stdout = False
retrack = False


def get_output_file(input_filename):
    return os.path.splitext(input_filename)[0]


def init_worker(c, cache, to_sout, re):
    global config
    global cache_to_disk
    global retrack
    global to_stdout
    config = c
    cache_to_disk = cache
    to_stdout = to_sout
    retrack = re


def extract_thread(filename):
    """
    Process a file extracting tracks and identifying them.
    :param filename: filename to process
    :param enable_preview: if true an MPEG preview file is created.
    """
    global config
    global cache_to_disk
    global to_stdout
    global retrack
    extract_file(filename, config, cache_to_disk, retrack=retrack, to_stdout=to_stdout)


def extract_file(filename, config, cache_to_disk, retrack=False, to_stdout=False):

    filename = Path(filename)
    if not filename.is_file():
        raise Exception("File {} not found.".format(filename))
    logging.info("Tracking %s", filename)
    previewer = Previewer.create_if_required(config, config.classify.preview)
    extension = filename.suffix
    if extension == ".cptv":
        track_extractor = ClipTrackExtractor(
            config.tracking,
            config.use_opt_flow,
            cache_to_disk,
            verbose=config.verbose,
        )
        logging.info("Using cptv extractor")

    else:
        track_extractor = IRTrackExtractor(
            config.tracking,
            cache_to_disk,
            verbose=config.verbose,
            keep_frames=True,
            # False if previewer is None else True,
            tracking_alg="subsense",
        )

        logging.info("Using ir extractor")
    clip = Clip(track_extractor.config, filename)
    if extension == ".cptv":
        clip.frames_per_second = 9
    else:
        clip.frames_per_second = 10
    existing_metadata = None
    if filename.with_suffix(".txt").exists():
        existing_metadata = tools.load_clip_metadata(filename.with_suffix(".txt"))

    if retrack:
        logging.info("Retracking")
        clip.load_metadata(existing_metadata)

    success = track_extractor.parse_clip(clip)

    # clip, success, tracking_time = extract_tracks(filename, config, cache_to_disk)
    if not success:
        logging.error("Could not parse %s", filename)
        return

    if retrack:
        for track in clip.tracks:
            track.trim()
            track.set_end_s(clip.frames_per_second)
    meta_filename = filename.with_suffix(".txt")

    if previewer:
        mpeg_filename = filename.parent / f"{filename.stem}-tracking.mp4"
        previewer.export_clip_preview(mpeg_filename, clip)
    logging.info("saving meta data %s", meta_filename)

    save_metadata(
        existing_metadata, filename, meta_filename, clip, track_extractor, to_stdout
    )
    if cache_to_disk:
        clip.frame_buffer.remove_cache()

    return clip, track_extractor


def save_metadata(
    existing_metadata, filename, meta_filename, clip, track_extractor, to_stdout=False
):
    metadata = clip.get_metadata()
    for i, track in enumerate(clip.tracks):
        best_thumb, best_score = get_thumbnail_info(clip, track)
        if best_thumb is None:
            metadata["tracks"][i]["thumbnail"] = None
            continue
        thumbnail_info = {
            "region": best_thumb.region,
            "contours": best_thumb.contours,
            "median_diff": best_thumb.median_diff,
            "score": round(best_score),
        }
        metadata["tracks"][i]["thumbnail"] = thumbnail_info
    if len(clip.tracks) == 0:
        # if no tracks choose a clip thumb
        region = best_trackless_thumb(clip)
        metadata["thumbnail_region"] = region
    metadata["source"] = str(filename)
    metadata["tracking_time"] = round(track_extractor.tracking_time, 1)
    metadata["algorithm"] = {}
    metadata["algorithm"]["tracker_version"] = track_extractor.tracker_version
    metadata["algorithm"]["tracker_config"] = track_extractor.config.as_dict()

    if existing_metadata is not None:
        # merge new metadata with old, think tracks is all that should be removed first
        if "tracks" in existing_metadata:
            del existing_metadata["tracks"]
        if "Tracks" in existing_metadata:
            del existing_metadata["Tracks"]

        existing_metadata.update(metadata)
        metadata = existing_metadata
    if to_stdout:
        print(json.dumps(metadata, cls=tools.CustomJSONEncoder))
    else:
        with open(meta_filename, "w") as f:
            json.dump(metadata, f, indent=4, cls=tools.CustomJSONEncoder)
