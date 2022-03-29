from multiprocessing import Pool

import cv2
import gc
import json
import logging
import os.path
import time

import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from load.irtrackextractor import IRTrackExtractor

from ml_tools import tools
from ml_tools.previewer import Previewer
from track.track import Track

from classify.thumbnail import get_thumbnail


class TrackExtractor:
    """Generate tracks for CPTV files."""

    def __init__(self, config, cache_to_disk=None):
        """Create an instance of a clip classifier"""

        self.config = config
        self.worker_threads = config.worker_threads

        if cache_to_disk is None:
            self.cache_to_disk = self.config.classify.cache_to_disk
        else:
            self.cache_to_disk = cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        #
        # self.track_extractor = IRTrackExtractor(
        #     self.config.tracking,
        #     self.config.use_opt_flow,
        #     self.cache_to_disk,
        #     high_quality_optical_flow=self.config.tracking.high_quality_optical_flow,
        #     verbose=self.config.verbose,
        #     keep_frames=False if self.previewer is None else True,
        # )

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
        if os.path.isfile(base):
            init_worker(self.config, self.cache_to_disk)
            extract_file(base)
            return
        data = []
        for folder_path, _, files in os.walk(base):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".avi", ".cptv"]:
                    full_path = os.path.join(folder_path, name)
                    data.append(full_path)
        print("running pool with", len(data))
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


def extract_file(filename):
    """
    Process a file extracting tracks and identifying them.
    :param filename: filename to process
    :param enable_preview: if true an MPEG preview file is created.
    """
    global config
    global cache_to_disk
    print("extracting", filename, config is not None, cache_to_disk)
    if not os.path.exists(filename):
        raise Exception("File {} not found.".format(filename))
    logging.info("Processing file '{}'".format(filename))
    previewer = Previewer.create_if_required(config, config.classify.preview)

    # GP RMEOVE JUST FOR TESTING
    out_file = get_output_file(filename)
    destination_folder = os.path.dirname(out_file)
    if not os.path.exists(destination_folder):
        logging.info("Creating folder {}".format(destination_folder))
        os.makedirs(destination_folder)
    meta_filename = out_file + ".txt"

    if not os.path.exists(meta_filename):
        logging.info("ALREADY Tracking %s", meta_filename)
        return

    track_extractor = IRTrackExtractor(
        config.tracking,
        config.use_opt_flow,
        cache_to_disk,
        high_quality_optical_flow=config.tracking.high_quality_optical_flow,
        verbose=config.verbose,
        keep_frames=False if previewer is None else True,
    )
    start = time.time()
    clip = Clip(config.tracking, filename)
    success = track_extractor.parse_clip(clip)

    # clip, success, tracking_time = extract_tracks(filename, config, cache_to_disk)
    if not success:
        logging.error("Could not parse %s", filename)
        return
    out_file = get_output_file(filename)
    destination_folder = os.path.dirname(out_file)
    if not os.path.exists(destination_folder):
        logging.info("Creating folder {}".format(destination_folder))
        os.makedirs(destination_folder)
    meta_filename = out_file + ".txt"

    if previewer:
        base_name = os.path.basename(out_file)
        mpeg_filename = destination_folder + "/" + base_name + "-tracking.avi"

        previewer.export_clip_preview(mpeg_filename, clip)
        # return
        # logging.info("Exporting preview to '{}'".format(mpeg_filename))
        # out = cv2.VideoWriter(
        #     mpeg_filename,
        #     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        #     10,
        #     (clip.res_x, clip.res_y),
        #     0,
        # )
        # for frame_number, frame in enumerate(clip.frame_buffer):
        #     for index, track in enumerate(clip.tracks):
        #         frame_offset = frame_number - track.start_frame
        #
        #         if frame_offset >= 0 and frame_offset < len(track.bounds_history):
        #             region = track.bounds_history[frame_offset]
        #             cv2.rectangle(
        #                 frame.thermal,
        #                 (region.left, region.top),
        #                 (region.right, region.bottom),
        #                 (0, 255, 0),
        #                 3,
        #             )
        #             print(
        #                 "region region",
        #                 region,
        #                 region.frame_number,
        #                 " for rame X",
        #                 frame_number,
        #             )
        #     out.write(frame.thermal)

        #
        # cv2.imshow("detected.png", np.uint8(frame.thermal))
        # cv2.waitKey(100)
        # self.previewer.export_clip_preview(mpeg_filename, clip)
    logging.info("saving meta data %s", meta_filename)
    # out.release()

    # cv2.destroyAllWindows()

    save_metadata(
        filename,
        meta_filename,
        clip,
        track_extractor,
    )
    if cache_to_disk:
        clip.frame_buffer.remove_cache()


#
# def extract_tracks(filename, config, cache_to_disk):
#     if not os.path.exists(filename):
#         raise Exception("File {} not found.".format(filename))
#     logging.info("Processing file '{}'".format(filename))
#
#     track_extractor = IRTrackExtractor(
#         config.tracking,
#         config.use_opt_flow,
#         cache_to_disk,
#         high_quality_optical_flow=config.tracking.high_quality_optical_flow,
#         verbose=config.verbose,
#         keep_frames=False if previewer is None else True,
#     )
#     start = time.time()
#     clip = Clip(config.tracking, filename)
#     success = track_extractor.parse_clip(clip)
#     return clip, success, track_extractor.tracking_time


def save_metadata(
    filename,
    meta_filename,
    clip,
    track_extractor,
):

    # record results in text file.
    save_file = clip.get_metadata()
    save_file["source"] = filename

    save_file["tracking_time"] = round(track_extractor.tracking_time, 1)
    save_file["algorithm"] = {}
    save_file["algorithm"]["tracker_version"] = track_extractor.tracker_version
    save_file["algorithm"]["tracker_config"] = self.config.tracking.as_dict()

    if self.config.classify.meta_to_stdout:
        print(json.dumps(save_file, cls=tools.CustomJSONEncoder))
    else:
        with open(meta_filename, "w") as f:
            json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)
