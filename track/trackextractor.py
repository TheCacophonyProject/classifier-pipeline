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

from classify.thumbnail import get_thumbnail_info, thumbnail_debug, best_trackless_thumb


class TrackExtractor:
    """Generate tracks for CPTV files."""

    def __init__(self, config, cache_to_disk=None, retrack=False):
        """Create an instance of a clip classifier"""

        self.config = config
        self.worker_threads = config.worker_threads
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

    def extract(self, base):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        base = Path(base)
        if not base.exists():
            return
        if base.is_file():
            init_worker(self.config, self.cache_to_disk)
            extract_file(base, self.retrack)
            return
        data = []
        for folder_path, _, files in os.walk(base):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".avi", ".cptv"]:
                    full_path = os.path.join(folder_path, name)
                    data.append(full_path, self.retrack)
        with Pool(
            self.worker_threads, init_worker, (self.config, self.cache_to_disk)
        ) as pool:
            pool.map(extract_file, data)


config = None
cache_to_disk = None


def get_output_file(input_filename):
    return os.path.splitext(input_filename)[0]


def init_worker(c, cache):
    global config
    global cache_to_disk
    config = c
    cache_to_disk = cache


def extract_file(filename, retrack=False):
    """
    Process a file extracting tracks and identifying them.
    :param filename: filename to process
    :param enable_preview: if true an MPEG preview file is created.
    """
    filename = Path(filename)
    global config
    global cache_to_disk
    if not filename.is_file():
        raise Exception("File {} not found.".format(filename))
    logging.info("Processing file '{}'".format(filename))
    previewer = Previewer.create_if_required(config, config.classify.preview)
    extension = filename.suffix
    if extension == ".cptv":
        track_extractor = ClipTrackExtractor(
            config.tracking,
            config.use_opt_flow,
            cache_to_disk,
            high_quality_optical_flow=config.tracking[
                "thermal"
            ].high_quality_optical_flow,
            verbose=config.verbose,
            keep_frames=True,
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

    if retrack:
        logging.info("Retracking")
        metadata = tools.load_clip_metadata(filename.with_suffix(".txt"))
        clip.load_metadata(metadata)
    start = time.time()
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

    save_metadata(filename, meta_filename, clip, track_extractor, config)
    if cache_to_disk:
        clip.frame_buffer.remove_cache()


def save_metadata(filename, meta_filename, clip, track_extractor, config):
    # record results in text file.
    metadata = clip.get_metadata()
    tt = sorted(clip.tracks, key=lambda x: x.get_id())
    for i, track in enumerate(tt):
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
    if config.classify.meta_to_stdout:
        print(json.dumps(metadata, cls=tools.CustomJSONEncoder))
    else:
        with open(meta_filename, "w") as f:
            json.dump(metadata, f, indent=4, cls=tools.CustomJSONEncoder)
