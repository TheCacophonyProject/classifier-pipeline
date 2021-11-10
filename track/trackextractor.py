import gc
import json
import logging
import os.path
import time

import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools import tools
from ml_tools.previewer import Previewer
from track.track import Track

from classify.thumbnail import get_thumbnail


class TrackExtractor:
    """Generate tracks for CPTV files."""

    def __init__(self, config, cache_to_disk=None):
        """Create an instance of a clip classifier"""

        self.config = config

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        if cache_to_disk is None:
            self.cache_to_disk = self.config.classify.cache_to_disk
        else:
            self.cache_to_disk = cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            self.cache_to_disk,
            high_quality_optical_flow=self.config.tracking.high_quality_optical_flow,
            verbose=self.config.verbose,
            keep_frames=False if self.previewer is None else True,
        )

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

    def get_output_file(self, input_filename):
        return os.path.splitext(input_filename)[0]

    def extract(self, base):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        if os.path.splitext(base)[1] == ".cptv":
            self.extract_file(base)
            return
        for folder_path, _, files in os.walk(base):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    self.extract_file(full_path)

    def extract_file(self, filename):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        clip = self.extract_tracks(filename)

        out_file = self.get_output_file(filename)
        destination_folder = os.path.dirname(out_file)
        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)
        meta_filename = out_file + ".txt"

        if self.previewer:
            mpeg_filename = out_file + ".mp4"
            logging.info("Exporting preview to '{}'".format(mpeg_filename))
            self.previewer.export_clip_preview(mpeg_filename, clip)
        logging.info("saving meta data %s", meta_filename)
        self.save_metadata(
            filename,
            meta_filename,
            clip,
            self.track_extractor.tracking_time,
        )

    def extract_tracks(self, filename):
        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))
        logging.info("Processing file '{}'".format(filename))

        start = time.time()
        clip = Clip(self.config.tracking, filename)
        self.track_extractor.parse_clip(clip)
        return clip

    def save_metadata(
        self,
        filename,
        meta_filename,
        clip,
        tracking_time,
    ):

        # record results in text file.
        save_file = {}
        save_file["source"] = filename
        if clip.camera_model:
            save_file["camera_model"] = clip.camera_model
        save_file["background_thresh"] = clip.background_thresh
        start, end = clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["tracking_time"] = round(tracking_time, 1)
        save_file["algorithm"] = {}
        save_file["algorithm"]["tracker_version"] = ClipTrackExtractor.VERSION
        save_file["algorithm"]["tracker_config"] = self.config.tracking.as_dict()

        tracks = []
        for track in clip.tracks:
            track_info = track.get_metadata()
            tracks.append(track_info)
        save_file["tracks"] = tracks

        if self.config.classify.meta_to_stdout:
            print(json.dumps(save_file, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)
        if self.cache_to_disk:
            clip.frame_buffer.remove_cache()
