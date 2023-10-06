"" """
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import logging
import time
from multiprocessing import Process, Queue
import traceback
from ml_tools import tools

from ml_tools.previewer import Previewer
from ml_tools.trackdatabase import TrackDatabase
from .clip import Clip
from .irtrackextractor import IRTrackExtractor
from .cliptrackextractor import ClipTrackExtractor
from ml_tools.imageprocessing import clear_frame
from track.track import Track
import numpy as np
import json


def process_job(loader, queue):
    i = 0
    while True:
        i += 1
        filename, clip_id = queue.get()
        try:
            if filename == "DONE":
                break
            else:
                loader.process_file(str(filename), clip_id)
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s %s", filename, e)
            traceback.print_exc()


class ClipLoader:
    def __init__(self, config, reprocess=False):
        self.config = config
        os.makedirs(self.config.tracks_folder, mode=0o775, exist_ok=True)
        self.database = TrackDatabase(
            os.path.join(self.config.tracks_folder, "dataset.hdf5")
        )
        self.reprocess = reprocess
        self.compression = (
            tools.gzip_compression if self.config.load.enable_compression else None
        )
        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads
        self.previewer = Previewer.create_if_required(config, config.load.preview)

    def process_all(self, root):
        clip_id = self.database.get_unique_clip_id()
        job_queue = Queue()
        processes = []
        for i in range(max(1, self.workers_threads)):
            p = Process(
                target=process_job,
                args=(self, job_queue),
            )
            processes.append(p)
            p.start()
        if root is None:
            root = self.config.source_folder
        file_paths = []
        for folder_path, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".avi", ".cptv"]:
                    full_path = os.path.join(folder_path, name)
                    file_paths.append(full_path)
        # allows us know the order of processing
        file_paths.sort()
        for file_path in file_paths:
            job_queue.put((file_path, clip_id))
            clip_id += 1

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put(("DONE", 0))
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()

    def _get_dest_folder(self, filename):
        return os.path.join(self.config.tracks_folder, get_distributed_folder(filename))

    def _export_tracks(self, full_path, clip):
        """
        Writes tracks to a track database.
        :param database: database to write track to.
        """
        # overwrite any old clips.
        # Note: we do this even if there are no tracks so there there will be a blank clip entry as a record
        # that we have processed it.
        self.database.create_clip(clip)
        for track in clip.tracks:
            start_time, end_time = clip.start_and_end_time_absolute(
                track.start_s, track.end_s
            )
            original_thermal = []
            cropped_data = []
            for region in track.bounds_history:
                frame = clip.frame_buffer.get_frame(region.frame_number)
                original_thermal.append(frame.thermal)
                cropped = frame.crop_by_region(region)
                # zero out the filtered channel
                if not self.config.load.include_filtered_channel:
                    cropped.filtered = np.zeros(cropped.thermal.shape)
                cropped_data.append(cropped)

            # sample_frames = get_sample_frames(
            #     clip.ffc_frames,
            #     [bounds.mass for bounds in track.bounds_history],
            #     self.config.build.segment_min_avg_mass,
            #     cropped_data,
            # )
            try:
                self.database.add_track(
                    clip.get_id(),
                    track,
                    cropped_data,
                    opts=self.compression,
                    original_thermal=original_thermal,
                    start_time=start_time,
                    end_time=end_time,
                )
            except:
                self.database.remove_track(clip.get_id(), track.get_id())
                logging.error(
                    "Error adding track %s - %s",
                    clip.get_id(),
                    track.get_id(),
                    exc_info=True,
                )
        self.database.finished_processing(clip.get_id())

    def _filter_clip_tracks(self, clip_metadata, min_confidence):
        """
        Removes track metadata for tracks which are invalid. Tracks are invalid
        if they aren't confident or they are in the excluded_tags list.
        Returns valid tracks
        """

        tracks_meta = clip_metadata.get("Tracks")
        if tracks_meta is None:
            tracks_meta = clip_metadata.get("tracks", [])
        valid_tracks = [
            track
            for track in tracks_meta
            if self._track_meta_is_valid(track, min_confidence)
        ]

        clip_metadata["Tracks"] = valid_tracks
        return valid_tracks

    def _track_meta_is_valid(self, track_meta, min_confidence):
        """
        Tracks are valid if their confidence meets the threshold and they are
        not in the excluded_tags list, defined in the config.
        """
        track_tags = []
        if "tags" in track_meta:
            track_tags = track_meta["tags"]
        excluded_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False)
            and tag.get("what") in self.config.load.excluded_tags
        ]
        if len(excluded_tags) > 0:
            return False

        track_tag = Track.get_best_human_tag(
            track_tags, self.config.load.tag_precedence, min_confidence
        )

        if track_tag is None:
            return False
        tag = track_tag.get("what")
        confidence = track_tag.get("confidence", 0)
        return tag and tag not in excluded_tags and confidence >= min_confidence

    def get_track_extractor(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == ".cptv":
            track_extractor = ClipTrackExtractor(
                self.config.tracking,
                self.config.use_opt_flow
                or self.config.load.preview == Previewer.PREVIEW_TRACKING,
                self.config.load.cache_to_disk,
                high_quality_optical_flow=self.config.load.high_quality_optical_flow,
                verbose=self.config.verbose,
                do_tracking=False,
            )
        else:
            track_extractor = IRTrackExtractor(
                self.config.tracking,
                self.config.load.cache_to_disk,
                verbose=self.config.verbose,
                do_tracking=True,
            )
        return track_extractor

    def process_file(self, filename, clip_id=None, classifier=None):
        start = time.time()
        base_filename = os.path.splitext(os.path.basename(filename))[0]

        logging.info(f"processing %s", filename)

        track_extractor = self.get_track_extractor(filename)

        destination_folder = self._get_dest_folder(base_filename)
        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")

        # read metadata
        metadata_filename = os.path.join(
            os.path.dirname(filename), base_filename + ".txt"
        )

        if not os.path.isfile(metadata_filename):
            logging.error("No meta data found for %s", metadata_filename)
            return

        metadata = tools.load_clip_metadata(metadata_filename)
        if "id" not in metadata and clip_id is not None:
            metadata["id"] = clip_id
            logging.info("Using clip id %s", clip_id)
            with open(metadata_filename, "w") as f:
                json.dump(metadata, f, indent=4, cls=tools.CustomJSONEncoder)
        if not self.reprocess and self.database.has_clip(str(metadata["id"])):
            logging.warning("Already loaded %s", filename)
            return
        valid_tracks = self._filter_clip_tracks(
            metadata, track_extractor.config.min_tag_confidence
        )

        if not valid_tracks or len(valid_tracks) == 0:
            logging.error("No valid track data found for %s", filename)
            return
        clip = Clip(track_extractor.config, filename)
        clip.load_metadata(
            metadata,
            self.config.load.tag_precedence,
        )
        tracker_version = metadata.get("tracker_version", 0)
        process_background = tracker_version < 10
        logging.debug(
            "Processing background? %s tracker_version %s",
            process_background,
            tracker_version,
        )
        if not track_extractor.parse_clip(clip, process_background=process_background):
            logging.error("No valid clip found for %s", filename)
            return

        self._export_tracks(filename, clip)

        # write a preview
        if self.previewer:
            os.makedirs(destination_folder, mode=0o775, exist_ok=True)

            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, clip)
            self.previewer.export_clip_preview(preview_filename, clip)

        if self.config.verbose:
            num_frames = len(clip.frame_buffer.frames)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self._log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(clip.tracks), num_frames, ms_per_frame
                )
            )

    def _log_message(self, message):
        """Record message in stdout.  Will be printed if verbose is enabled."""
        # note, python has really good logging... I should probably make use of this.
        if self.config.verbose:
            logging.info(message)


def get_distributed_folder(name, num_folders=256, seed=31):
    """Creates a hash of the name then returns the modulo num_folders"""
    str_bytes = str.encode(name)
    hash_code = 0
    for byte in str_bytes:
        hash_code = hash_code * seed + byte
    return "{:02x}".format(hash_code % num_folders)


def get_sample_frames(ffc_frames, mass_history, min_mass=None, frame_data=None):
    clear_frames = []
    lower_mass = np.percentile(mass_history, q=25)
    upper_mass = np.percentile(mass_history, q=75)
    for i, mass in enumerate(mass_history):
        if i in ffc_frames:
            continue
        if (
            min_mass is None
            or mass >= min_mass
            and mass >= lower_mass
            and mass <= upper_mass
        ):
            if frame_data is not None:
                if not clear_frame(frame_data[i]):
                    continue
            clear_frames.append(i)
    return clear_frames
