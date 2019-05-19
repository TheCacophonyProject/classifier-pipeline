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
from ml_tools import trackdatabase
from ml_tools.previewer import Previewer
from .clip import Clip


def init_workers(lock):
    """ Initialise worker by setting the trackdatabase lock. """
    trackdatabase.HDF5_LOCK = lock


def process_job(job):
    job[0].process_file(job[1])


class ClipLoader:
    def __init__(self, config, tracker_config):

        self.config = config
        os.makedirs(self.config.tracks_folder, mode=0o775, exist_ok=True)
        self.database = TrackDatabase(
            os.path.join(self.config.tracks_folder, "dataset.hdf5")
        )

        self.enable_track_output = tracker_config.enable_track_output
        self.worker_pool_init = init_workers
        self.track_config = tracker_config
        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads

        self.previewer = Previewer.create_if_required(config, config.extract.preview)

    def process_all(self, root=None):
        if root is None:
            root = self.config.source_folder

        jobs = []
        for folder_path, _, files in os.walk(root):
            if os.path.basename(folder_path) in self.config.excluded_folders:
                return
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    jobs.append((self, full_path))

        self.process_jobs(jobs)

    def process_jobs(self, jobs):
        if self.workers_threads == 0:
            for job in jobs:
                process_job(job)
        else:
            pool = multiprocessing.Pool(
                self.workers_threads,
                initializer=self.worker_pool_init,
                initargs=init_workers,
            )
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

    def get_dest_folder(self, filename):
        return self.config.tracks_folder

    def export_tracks(self, full_path, clip):
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

            self.database.add_track(
                clip.get_id(),
                track,
                # opts=self.compression,
                start_time=start_time,
                end_time=end_time,
            )

    def process_file(self, filename):
        # tag = kwargs["tag"]
        start = time.time()
        base_filename = os.path.splitext(os.path.basename(filename))[0]

        logging.info(f"processing %s", filename)

        destination_folder = self.get_dest_folder(filename)
        os.makedirs(destination_folder, mode=0o775, exist_ok=True)

        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")

        clip = Clip(self.track_config)
        clip.load_cptv(filename)
        # read metadata
        metadata_filename = os.path.join(
            os.path.dirname(filename), base_filename + ".txt"
        )
        if os.path.isfile(metadata_filename):
            metadata = tools.load_clip_metadata(metadata_filename)
            clip.parse_clip(metadata, self.config.extract.include_filtered_channel)
        else:
            logging.error("No meta data found for %s", metadata_filename)
            return

        if self.enable_track_output:
            self.export_tracks(filename, clip)

        # write a preview
        if self.previewer:
            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, clip)
            self.previewer.export_clip_preview(preview_filename, clip)

        if self.track_config.verbose:
            num_frames = len(clip.frame_buffer.thermal)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self.log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(tracker.tracks), num_frames, ms_per_frame
                )
            )

    def log_message(self, message):
        """ Record message in stdout.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.track_config.verbose:
            logging.info(message)
