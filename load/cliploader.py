import os
from track.track import Track
from track.region import Region

def init_workers(lock):
    """ Initialise worker by setting the trackdatabase lock. """
    trackdatabase.HDF5_LOCK = lock

class Trackloader:

    def __init__(self, config, tracker_config):

        os.makedirs(self.config.tracks_folder, mode=0o775, exist_ok=True)
        self.database = TrackDatabase(
            os.path.join(self.config.tracks_folder, "dataset.hdf5")
        )

        self.worker_pool_init = init_workers

        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads


    def process_all(self, root=None):
        if root is None:
            root = self.config.source_folder

        jobs = []
        for folder_path, folders, files in os.walk(root):
            os.basenamefolder_path
            if os.path.basename(folder_path) in self.config.excluded_folders:
                return;
           for name in files:
                if os.path.splitext(name)[1]== ".cptv":
                    full_path = os.path.join(folder_path, file_name)
                    jobs.append((self, full_path))

        process_jobs(jobs)

    def process_jobs(self, jobs):
        if self.workers_threads == 0:
            # just process the jobs in the main thread
            for job in jobs:
                load_file(job)
        else:
            # send the jobs to a worker pool
            pool = multiprocessing.Pool(
                self.workers_threads,
                initializer=self.worker_pool_init,
                initargs=worker_pool_args,
            )
            try:
                # see https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
                pool.map(load_file, jobs, chunksize=1)
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

    def get_dest_folder(filename):
        return self.config.tracks_folder

    
    def export_tracks(
        self, full_path, clip, metadata
    ):
        """
        Writes tracks to a track database.
        :param database: database to write track to.
        """

        clip_id = os.path.basename(full_path)

        # overwrite any old clips.
        # Note: we do this even if there are no tracks so there there will be a blank clip entry as a record
        # that we have processed it.
        database.create_clip(clip_id, tracker)

        if len(tracker.tracks) == 0:
            return

        clip.generate_optical_flow()
        tracks_meta = metadata["tracks"]
        # get track data
        for track_meta in tracks_meta:
            track = Track()
            track.load_meta(track_meta)
            track_data = []
            start_s = track["start_s"]
            end_s = track["end_s"]
            regions=track_data["positions"]

            for region in regions:
                bounds = region_from_json(region[1])
                track.add_frame(bounds)
                frame_number = round(region[0] * clip.FRAMES_PER_SECOND) 

                channels = clip.get_frame_channels(bounds, frame_number)

                # zero out the filtered channel
                if not self.config.extract.include_filtered_channel:
                    channels[TrackChannels.filtered] = 0
                track_data.append(channels)
                
            start_time, end_time = clip.start_and_end_time_absolute(track)
            database.add_track(
                clip_id,
                track_id,
                track_data,
                track,
                opts=self.compression,
                start_time=start_time,
                end_time=end_time,
            )


    def load_file(self, filename):
        tag = kwargs["tag"]

        base_filename = os.path.splitext(os.path.basename(full_path))[0]

        logging.info(f"processing %s", filename)

        destination_folder = get_dest_folder(filename)
        os.makedirs(destination_folder, mode=0o775, exist_ok=True)

        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")


        clip = Clip()
        clip.load_cptv(filename)
        clip.parse_clip()
        # read metadata
        metadata_filename = os.path.basename(full_path) + ".txt"
        if os.path.isfile(metadata_filename):
            metadata = tools.load_clip_metadata(meta_data_filename)
        
         if self.enable_track_output:
            self.export_tracks(full_path, clip, metadata, self.database)






        # write a preview
        if self.previewer:
            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, tracker)
            self.previewer.export_clip_preview(preview_filename, tracker)

        if self.tracker_config.verbose:
            num_frames = len(tracker.frame_buffer.thermal)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self.log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(tracker.tracks), num_frames, ms_per_frame
                )
            )

        return tracker


            tags = set(
                [
                    tag["animal"]
                    for tag in meta_data["Tags"]
                    if "automatic" not in tag or not tag["automatic"]
                ]
            )

            # we can only handle one tagged animal at a time here.
            if len(tags) == 0:
                logging.warning(" - no tags in cptv files, ignoring.")
                return

            if len(tags) >= 2:
                # make sure all tags are the same
                logging.warning(" - mixed tags, can not process: %s", tags)
                return

            tracker.stats["confidence"] = meta_data["Tags"][0].get("confidence", 0.0)
            tracker.stats["trap"] = meta_data["Tags"][0].get("trap", "none")
            tracker.stats["event"] = meta_data["Tags"][0].get("event", "none")

            # clips tagged with false-positive sometimes come through with a null confidence rating
            # so we set it to 0.8 here.
            if (
                tracker.stats["event"] in ["false-positive", "false positive"]
                and tracker.stats["confidence"] is None
            ):
                tracker.stats["confidence"] = 0.8

            tracker.stats["cptv_metadata"] = meta_data
        else:
            self.log_warning(
                " - Warning: no tag metadata found for file - cannot use for machine learning."
            )

        start = time.time()

        # save some additional stats
        tracker.stats["version"] = TrackExtractor.VERSION

        tracker.load(full_path)

        if not tracker.extract_tracks():
            # this happens if the tracker rejected the video for some reason (i.e. too hot, or not static background).
            # we still need to make a record that we looked at it though.
            self.database.create_clip(os.path.basename(full_path), tracker)
            logging.warning(" - skipped (%s)", tracker.reject_reason)
            return tracker

        # assign each track the correct tag
        for track in tracker.tracks:
            track.tag = tag

        if self.enable_track_output:
            self.export_tracks(full_path, tracker, self.database)

        # write a preview
        if self.previewer:
            preview_filename = base_filename + "-preview" + ".mp4"
            preview_filename = os.path.join(destination_folder, preview_filename)
            self.previewer.create_individual_track_previews(preview_filename, tracker)
            self.previewer.export_clip_preview(preview_filename, tracker)

        if self.tracker_config.verbose:
            num_frames = len(tracker.frame_buffer.thermal)
            ms_per_frame = (time.time() - start) * 1000 / max(1, num_frames)
            self.log_message(
                "Tracks {}.  Frames: {}, Took {:.1f}ms per frame".format(
                    len(tracker.tracks), num_frames, ms_per_frame
                )
            )

        return tracker