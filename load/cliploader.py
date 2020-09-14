"""
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
import multiprocessing
import time

from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase

from ml_tools.previewer import Previewer
from .clip import Clip
from .cliptrackextractor import ClipTrackExtractor
from track.track import Track, TrackChannels


def process_job(job):
    job[0].process_file(job[1])


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
        self.track_config = config.tracking
        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads
        self.previewer = Previewer.create_if_required(config, config.load.preview)
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.load.preview == Previewer.PREVIEW_TRACKING,
            self.config.load.cache_to_disk,
        )

    def process_all(self, root=None, checkpoint=None):
        if root is None:
            root = self.config.source_folder

        jobs = []
        for folder_path, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    jobs.append((self, full_path))

        jobs.sort(key=lambda x: x[1])

        self._process_jobs(jobs)

    def _process_jobs(self, jobs):
        if self.workers_threads == 0:
            for job in jobs:
                process_job(job)
        else:
            pool = multiprocessing.Pool(self.workers_threads)
            try:
                pool.map(process_job, jobs, chunksize=1)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                pool.terminate()
                exit()
            except Exception:
                logging.exception("Error processing files")
            else:
                pool.close()

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
            track_data = []
            for region in track.bounds_history:
                frame = clip.frame_buffer.get_frame(region.frame_number)
                frame = track.crop_by_region(frame, region, filter_mask_by_region=False)
                # zero out the filtered channel
                if not self.config.load.include_filtered_channel:
                    frame[TrackChannels.filtered] = 0
                track_data.append(frame)

            self.database.add_track(
                clip.get_id(),
                track,
                track_data,
                opts=self.compression,
                start_time=start_time,
                end_time=end_time,
            )

    def _filter_clip_tracks(self, clip_metadata):
        """
        Removes track metadata for tracks which are invalid. Tracks are invalid
        if they aren't confident or they are in the excluded_tags list.
        Returns valid tracks
        """

        tracks_meta = clip_metadata.get("Tracks", [])
        valid_tracks = [
            track for track in tracks_meta if self._track_meta_is_valid(track)
        ]
        clip_metadata["Tracks"] = valid_tracks
        return valid_tracks

    def _track_meta_is_valid(self, track_meta):
        """
        Tracks are valid if their confidence meets the threshold and they are
        not in the excluded_tags list, defined in the config.
        """
        min_confidence = self.track_config.min_tag_confidence
        excluded_tags = self.config.excluded_tags
        track_data = track_meta.get("data")
        if not track_data:
            return False

        track_tag = Track.get_best_human_tag(
            track_meta, self.config.load.tag_precedence, min_confidence
        )
        if track_tag is None:
            return False
        tag = track_tag.get("what")
        confidence = track_tag.get("confidence", 0)
        return tag and tag not in excluded_tags and confidence >= min_confidence

    def process_file(self, filename):
        start = time.time()
        base_filename = os.path.splitext(os.path.basename(filename))[0]

        logging.info(f"processing %s", filename)

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

        if not self.reprocess and self.database.has_clip(str(metadata["id"])):
            logging.warning("Already loaded %s", filename)
            return

        valid_tracks = self._filter_clip_tracks(metadata)
        if not valid_tracks:
            logging.error("No valid track data found for %s", filename)
            return

        clip = Clip(self.track_config, filename)
        clip.load_metadata(
            metadata,
            self.config.load.include_filtered_channel,
            self.config.load.tag_precedence,
        )

        if not self.track_extractor.parse_clip(clip):
            logging.error("No valid clip found for %s", filename)
            return

        # , self.config.load.cache_to_disk, self.config.use_opt_flow

        if self.track_config.enable_track_output:
            self._export_tracks(filename, clip)

        # write a preview
        if self.previewer:
            os.makedirs(destination_folder, mode=0o775, exist_ok=True)

            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, clip)
            self.previewer.export_clip_preview(preview_filename, clip)

        if self.track_config.verbose:
            num_frames = len(clip.frame_buffer.frames)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self._log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(clip.tracks), num_frames, ms_per_frame
                )
            )

    def _log_message(self, message):
        """ Record message in stdout.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.track_config.verbose:
            logging.info(message)


def get_distributed_folder(name, num_folders=256, seed=31):
    """Creates a hash of the name then returns the modulo num_folders"""
    str_bytes = str.encode(name)
    hash_code = 0
    for byte in str_bytes:
        hash_code = hash_code * seed + byte
    return "{:02x}".format(hash_code % num_folders)
