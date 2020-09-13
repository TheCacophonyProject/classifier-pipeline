import logging
import time
import os

from ml_tools.cptvfileprocessor import CPTVFileProcessor
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools
from ml_tools import trackdatabase

from track.track import TrackChannels
from ml_tools.tools import blosc_zstd
from ml_tools.previewer import Previewer
from track.trackextractor import TrackExtractor


class TrackerTestCase:
    def __init__(self):
        self.source = None
        self.tracks = []


class CPTVTrackExtractor(CPTVFileProcessor):
    """
    Handles extracting tracks from CPTV files.
    Maintains a database recording which files have already been processed, and some statistics parameters used
    during processing.
    """

    def __init__(self, config, tracker_config):

        CPTVFileProcessor.__init__(self, config, tracker_config)

        self.hints = {}
        self.enable_track_output = True
        self.compression = (
            blosc_zstd if self.config.extract.enable_compression else None
        )

        self.previewer = Previewer.create_if_required(config, config.extract.preview)

        # normally poor quality tracks are filtered out, enabling this will let them through.
        self.disable_track_filters = False
        # disables background subtraction
        self.disable_background_subtraction = False

        os.makedirs(self.config.tracks_folder, mode=0o775, exist_ok=True)
        self.database = TrackDatabase(
            os.path.join(self.config.tracks_folder, "dataset.hdf5")
        )

        # load hints.  Hints are a way to give extra information to the tracker when necessary.
        # if os.path.exists(config.extract.hints_file):
        if config.extract.hints_file:
            self.load_hints(config.extract.hints_file)

    def load_hints(self, filename):
        """ Read in hints file from given path.  If file is not found an empty hints dictionary set."""

        self.hints = {}

        if not os.path.exists(filename):
            logging.warning("Failed to load hints file: %s", filename)
            return

        f = open(filename)
        for line_number, line in enumerate(f):
            line = line.strip()
            # comments
            if line == "" or line[0] == "#":
                continue
            try:
                (filename, file_max_tracks) = line.split()[:2]
            except:
                raise Exception("Error on line {0}: {1}".format(line_number, line))
            self.hints[filename] = int(file_max_tracks)

    def process_all(self, root):
        if root is None:
            root = self.config.source_folder

        previous_filter_setting = self.disable_track_filters
        previous_background_setting = self.disable_background_subtraction
        for folder_root, folders, _ in os.walk(root):

            for folder in folders:
                if folder not in self.config.excluded_folders:
                    if folder.lower() == "false-positive":
                        self.disable_track_filters = True
                        self.disable_background_subtraction = True
                        logging.info("Turning Track filters OFF.")

                    self.process_folder(
                        os.path.join(folder_root, folder), tag=folder.lower()
                    )

                    if folder.lower() == "false-positive":
                        logging.info("Restoring Track filters.")
                        self.disable_track_filters = previous_filter_setting
                        self.disable_background_subtraction = (
                            previous_background_setting
                        )

    def clean_tag(self, tag):
        """
        Removes all clips with given tag.
        :param tag: label to remove
        """
        logging.info("removing tag: %s", tag)

        ids = self.database.get_all_track_ids()
        for (clip_id, track_number) in ids:
            if not self.database.has_clip(clip_id):
                continue
            meta = self.database.get_track_meta(clip_id, track_number)
            if meta["tag"] == tag:
                logging.info("removing: %s", clip_id)
                self.database.remove_clip(clip_id)

    def clean_all(self):
        """
        Checks if there are any clips in the database that are on the banned list.  Also makes sure no track has more
        tracks than specified in hints file.
        """

        for clip_id, max_tracks in self.hints.items():
            if self.database.has_clip(clip_id):
                if max_tracks == 0:
                    logging.info(" - removing banned clip %s", clip_id)
                    self.database.remove_clip(clip_id)
                else:
                    meta = self.database.get_clip_meta(clip_id)
                    if meta["tracks"] > max_tracks:
                        logging.info(" - removing out of date clip: %s", clip_id)
                        self.database.remove_clip(clip_id)

    def process_file(self, full_path, **kwargs):
        """
        Extract tracks from specific file, and assign given tag.
        :param full_path: path: path to CPTV file to be processed
        :param tag: the tag to assign all tracks from this CPTV files
        :returns the tracker object
        """

        tag = kwargs["tag"]

        base_filename = os.path.splitext(os.path.split(full_path)[1])[0]
        cptv_filename = base_filename + ".cptv"

        logging.info(f"processing %s", cptv_filename)

        destination_folder = os.path.join(self.config.tracks_folder, tag.lower())
        os.makedirs(destination_folder, mode=0o775, exist_ok=True)
        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")

        # read additional information from hints file
        if cptv_filename in self.hints:
            print(cptv_filename)
            logging.info(self.hints[cptv_filename])
            max_tracks = self.hints[cptv_filename]
            if max_tracks == 0:
                return
        else:
            max_tracks = self.config.tracking.max_tracks

        # load the track
        tracker = TrackExtractor(self.tracker_config)
        tracker.max_tracks = max_tracks
        tracker.tag = tag

        # by default we don't want to process the moving background images as it's too hard to get good tracks
        # without false-positives.
        tracker.reject_non_static_clips = True

        if self.disable_track_filters:
            tracker.track_min_delta = 0.0
            tracker.track_min_mass = 0.0
            tracker.track_min_offset = 0.0
            tracker.reject_non_static_clips = False

        if self.disable_background_subtraction:
            tracker.disable_background_subtraction = True

        # read metadata
        meta_data_filename = os.path.splitext(full_path)[0] + ".txt"
        if os.path.exists(meta_data_filename):

            meta_data = tools.load_clip_metadata(meta_data_filename)

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

    def export_tracks(
        self, full_path, tracks, tracker: TrackExtractor, database: TrackDatabase
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

        tracker.generate_optical_flow()

        # get track data
        for track_number, track in enumerate(tracker.tracks):
            track_data = []
            for i in range(len(track)):
                channels = tracker.get_track_channels(track, i)

                # zero out the filtered channel
                if not self.config.extract.include_filtered_channel:
                    channels[TrackChannels.filtered] = 0
                track_data.append(channels)
            track_id = track_number + 1
            start_time, end_time = tracker.start_and_end_time_absolute(track)
            database.add_track(
                clip_id,
                track_id,
                track_data,
                track,
                opts=self.compression,
                start_time=start_time,
                end_time=end_time,
            )

    def needs_processing(self, source_filename):
        """
        Returns if given source file needs processing or not
        :param source_filename:
        :return:
        """

        clip_id = os.path.basename(source_filename)

        if self.config.reprocess:
            return True

        return not self.database.has_clip(clip_id)

    def run_test(self, source_folder, test: TrackerTestCase):
        """ Runs a specific test case. """

        def are_similar(value, expected, relative_error=0.2, abs_error=2.0):
            """ Checks of value is similar to expected value. An expected value of 0 will always return true. """
            if expected == 0:
                return True
            return ((abs(value - expected) / expected) <= relative_error) or (
                abs(value - expected) <= abs_error
            )

        # find the file.  We looking in all the tag folder to make life simpler when creating the test file.
        source_file = tools.find_file(source_folder, test.source)

        # make sure we don't write to database
        self.enable_track_output = False

        if source_file is None:
            logging.warning(
                "Could not find %s in root folder %s", test.source, source_folder
            )
            return

        logging.info(source_file)
        tracker = self.process_file(source_file, tag="test")

        # read in stats files and see how we did
        if len(tracker.tracks) != len(test.tracks):
            logging.error(
                "%s Incorrect number of tracks, expected %s found %s",
                test.source,
                len(test.tracks),
                len(tracker.tracks),
            )
            return

        for test_result, (expected_duration, expected_movement) in zip(
            tracker.tracks, test.tracks
        ):

            track_stats = test_result.get_stats()

            if not are_similar(
                test_result.duration, expected_duration
            ) or not are_similar(track_stats.max_offset, expected_movement):
                logging.error(
                    "%s Track too dissimilar expected %s but found %s",
                    test.source,
                    (expected_duration, expected_movement),
                    (test_result.duration, track_stats.max_offset),
                )
            else:
                logging.info("%s passed", test.source)

    def run_tests(self, source_folder, tests_file):
        """ Processes file in test file and compares results to expected output. """

        # disable hints for tests
        self.hints = []

        tests = []
        test = None

        # # we need to make sure all tests are redone every time.
        # self.overwrite_mode = self.OM_ALL

        # load in the test data
        for line in open(tests_file, "r"):
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                continue

            if line.split()[0].lower() == "track":
                if test == None:
                    raise Exception("Can not have track before source file.")
                expected_length, expected_movement = [int(x) for x in line.split()[1:]]
                test.tracks.append((expected_length, expected_movement))
            else:
                test = TrackerTestCase()
                test.source = line
                tests.append(test)

        logging.info("Found %d test cases", len(tests))

        for test in tests:
            self.run_test(source_folder, test)
