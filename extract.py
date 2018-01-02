"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import cv2
from PIL import Image, ImageDraw
import numpy as np

from ml_tools import trackdatabase
from ml_tools.trackextractor import TrackExtractor
from ml_tools.cptvfileprocessor import CPTVFileProcessor
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools

import matplotlib.pyplot as plt
import os

import glob
import argparse
import time

# default base path to use if no source or destination folder are given.
DEFAULT_BASE_PATH = "c:\\cac"

EXCLUDED_FOLDERS = ['false-positive', 'other', 'unidentified', 'untagged']

class TrackerTestCase():
    def __init__(self):
        self.source = None
        self.tracks = []

def init_workers(lock):
    """ Initialise worker by setting the trackdatabase lock. """
    trackdatabase.hdf5_lock = lock

class CPTVTrackExtractor(CPTVFileProcessor):
    """
    Handles extracting tracks from CPTV files.
    Maintains a database recording which files have already been processed, and some statistics parameters used
    during processing.
    """

    # version number.  Recorded into stats file when a clip is processed.
    VERSION = 6

    def __init__(self, out_folder):

        CPTVFileProcessor.__init__(self)

        self.hints = {}
        self.colormap = plt.get_cmap('jet')
        self.verbose = False
        self.out_folder = out_folder
        self.overwrite_mode = CPTVTrackExtractor.OM_NONE
        self.enable_previews = False
        self.enable_track_output = True

        # normally poor quality tracks are filtered out, enabling this will let them through.
        self.disable_track_filters = False

        self.reduced_quality_optical_flow = False

        self.database = TrackDatabase(os.path.join(self.out_folder, 'dataset.hdf5'))

        self.worker_pool_init = init_workers

    def load_hints(self, filename):
        """ Read in hints file from given path.  If file is not found an empty hints dictionary set."""

        self.hints = {}

        if not os.path.exists(filename):
            return

        f = open(filename)
        for line_number, line in enumerate(f):
            line = line.strip()
            # comments
            if line == '' or line[0] == '#':
                continue
            try:
                (filename, file_max_tracks) = line.split()[:2]
            except:
                raise Exception("Error on line {0}: {1}".format(line_number, line))
            self.hints[filename] = int(file_max_tracks)

    def process_all(self, root):
        for root, folders, files in os.walk(root):
            for folder in folders:
                if folder not in EXCLUDED_FOLDERS:
                    self.process_folder(os.path.join(root,folder), tag=folder.lower(), worker_pool_args=(trackdatabase.hdf5_lock,))

    def process_file(self, full_path, **kwargs):
        """
        Extract tracks from specific file, and assign given tag.
        :param full_path: path: path to CPTV file to be processed
        :param tag: the tag to assign all tracks from this CPTV files
        :param create_preview_file: if enabled creates an MPEG preview file showing the tracking working.  This
            process can be quite time consuming.
        :returns the tracker object
        """

        tag = kwargs['tag']

        base_filename = os.path.splitext(os.path.split(full_path)[1])[0]
        cptv_filename = base_filename + '.cptv'
        preview_filename = base_filename + '-preview' + '.mp4'
        stats_filename = base_filename + '.txt'

        destination_folder = os.path.join(self.out_folder, tag.lower())

        stats_path_and_filename = os.path.join(destination_folder, stats_filename)

        # read additional information from hints file
        if cptv_filename in self.hints:
            max_tracks = self.hints[cptv_filename]
            if max_tracks == 0:
                return
        else:
            max_tracks = 10

        # make destination folder if required
        try:
            os.stat(destination_folder)
        except:
            self.log_message(" Making path " + destination_folder)
            os.mkdir(destination_folder)

        # check if we have already processed this file
        if self.needs_processing(stats_path_and_filename):
            print("Processing {0} [{1}]".format(cptv_filename, tag))
        else:
            return

        # delete any previous files
        tools.purge(destination_folder, base_filename + "*.mp4")

        # load the track
        tracker = TrackExtractor()
        tracker.max_tracks = max_tracks
        tracker.tag = tag
        tracker.verbose = self.verbose >= 2
        tracker.reduced_quality_optical_flow = self.reduced_quality_optical_flow

        if self.disable_track_filters:
            tracker.track_min_delta = 0.0
            tracker.track_min_mass = 0.0
            tracker.track_min_offset = 0.0

        # read metadata
        meta_data_filename = os.path.splitext(full_path)[0] + ".txt"
        if os.path.exists(meta_data_filename):

            meta_data = tools.load_clip_metadata(meta_data_filename)

            tag_count = len(meta_data['Tags'])

            # we can only handle one tagged animal at a time here.
            if tag_count == 0:
                print(" - Warning, no tags in cptv files, ignoring.")
                return

            if tag_count >= 2:
                # make sure all tags are the same
                tags = set([x['animal'] for x in meta_data['Tags']])
                if len(tags) >= 2:
                    print(" - Warning, mixed tags, can not process.",tags)
                    return

            tracker.stats['confidence'] = meta_data['Tags'][0].get('confidence',0.0)
            tracker.stats['trap'] = meta_data['Tags'][0].get('trap','none')
            tracker.stats['event'] = meta_data['Tags'][0].get('event','none')

            # clips tagged with false-positive sometimes come through with a null confidence rating
            # so we set it to 0.8 here.
            if tracker.stats['event'] in ['false-positive', 'false positive'] and tracker.stats['confidence'] is None:
                tracker.stats['confidence'] = 0.8

            tracker.stats['cptv_metadata'] = meta_data
        else:
            self.log_warning(" - Warning: no metadata found for file.")
            return

        start = time.time()

        # save some additional stats
        tracker.stats['version'] = CPTVTrackExtractor.VERSION

        tracker.load(full_path)

        if not tracker.extract_tracks():
            # this happens if the tracker rejected the video for some reason (i.e. too hot, or not static background).
            # we still need to make a record that we looked at it though.
            self.database.create_clip(os.path.basename(full_path), tracker)
            print(" - skipped ({})".format(tracker.reject_reason))
            return tracker

        # assign each track the correct tag
        for track in tracker.tracks:
            track.tag = tag

        if self.enable_track_output:
            tracker.export_tracks(self.database)

        # write a preview
        if self.enable_previews:
            self.export_mpeg_preview(os.path.join(destination_folder, preview_filename), tracker)

        time_per_frame = (time.time() - start) / len(tracker.frame_buffer)

        # time_stats = tracker.stats['time_per_frame']
        self.log_message(" -tracks: {} {:.1f}sec - Time per frame: {:.1f}ms]".format(
             len(tracker.tracks),
             sum(track.duration for track in tracker.tracks),
             time_per_frame * 1000
         ))

        return tracker

    def needs_processing(self, source_filename):
        """
        Returns if given source file needs processing or not
        :param source_filename:
        :return:
        """

        clip_id = os.path.basename(source_filename)

        if self.overwrite_mode == self.OM_ALL:
            return True

        return not self.database.has_clip(clip_id)

    def run_test(self, source_folder, test: TrackerTestCase):
        """ Runs a specific test case. """

        def are_similar(value, expected, relative_error = 0.2, abs_error = 2.0):
            """ Checks of value is similar to expected value. An expected value of 0 will always return true. """
            if expected == 0:
                return True
            return ((abs(value - expected) / expected) <= relative_error) or (abs(value - expected) <= abs_error)

        # find the file.  We looking in all the tag folder to make life simpler when creating the test file.
        source_file = tools.find_file(source_folder, test.source)

        # make sure we don't write to database
        self.enable_track_output = False

        if source_file is None:
            print("Could not find {0} in root folder {1}".format(test.source, source_folder))
            return

        print(source_file)
        tracker = self.process_file(source_file, tag='test')

        # read in stats files and see how we did
        if len(tracker.tracks) != len(test.tracks):
            print("[Fail] {0} Incorrect number of tracks, expected {1} found {2}".format(test.source, len(test.tracks), len(tracker.tracks)))
            return

        for test_result, (expected_duration, expected_movement) in zip(tracker.tracks, test.tracks):

            track_stats = test_result.get_stats()

            if not are_similar(test_result.duration, expected_duration) or not are_similar(track_stats.max_offset, expected_movement):
                print("[Fail] {0} Track too dissimilar expected {1} but found {2}".format(
                    test.source,
                    (expected_duration, expected_movement),
                    (test_result.duration, track_stats.max_offset)))
            else:
                print("[PASS] {0}".format(test.source))

    def export_track_mpeg_preview(self, filename_base, tracker: TrackExtractor):
        """
        Exports preview MPEG for a specific track
        :param filename_base:
        :param tracker:
        :param track:
        :return:
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 4.0

        for id, track in enumerate(tracker.tracks):
            video_frames = []
            for frame_number in range(len(track.bounds_history)):
                channels = tracker.get_track_channels(track, frame_number)
                img = tools.convert_heat_to_img(channels[0], self.colormap, tools.TEMPERATURE_MIN, tools.TEMPERATURE_MAX)
                img = img.resize((int(img.width * FRAME_SCALE), int(img.height * FRAME_SCALE)), Image.NEAREST)
                video_frames.append(np.asarray(img))

            tools.write_mpeg(filename_base+"-"+str(id+1)+".mpg", video_frames)

    def export_mpeg_preview(self, filename, tracker: TrackExtractor):
        """
        Exports tracking information preview to MPEG file.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 3.0

        video_frames = []
        track_colors = [(255,0,0),(0,255,0),(255,255,0),(128,255,255)]

        if not tracker.frame_buffer.has_flow:
            tracker.frame_buffer.generate_flow(tracker.opt_flow)

        self.export_track_mpeg_preview(os.path.splitext(filename)[0], tracker)

        for frame_number in range(len(tracker.frame_buffer.filtered)):
            thermal = tracker.frame_buffer.thermal[frame_number]
            filtered = tracker.frame_buffer.filtered[frame_number]
            mask = tracker.frame_buffer.mask[frame_number]
            flow = tracker.frame_buffer.flow[frame_number]
            regions = tracker.region_history[frame_number]

            # marked is an image with each pixel's value being the label, 0...n for n objects
            # I multiply it here, but really I should use a seperate color map for this.
            # maybe I could multiply it modulo, and offset by some amount?

            # This really should be using a pallete here, I multiply by 10000 to make sure the binary mask '1' values get set to the brightest color (which is about 4000)
            # here I map the flow magnitude [ranges in the single didgits) to a temperature in the display range.

            flow_magnitude = (flow[:,:,0]**2 + flow[:,:,1]**2) ** 0.5
            stacked = np.hstack((np.vstack((thermal, mask*10000)),np.vstack((3 * filtered + tools.TEMPERATURE_MIN, 200 * flow_magnitude + tools.TEMPERATURE_MIN))))

            img = tools.convert_heat_to_img(stacked, self.colormap, tools.TEMPERATURE_MIN, tools.TEMPERATURE_MAX)
            img = img.resize((int(img.width * FRAME_SCALE), int(img.height * FRAME_SCALE)), Image.NEAREST)
            draw = ImageDraw.Draw(img)

            # look for any regions of interest that occur on this frame
            for rect in regions:
                rect_points = [int(p * FRAME_SCALE) for p in [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom, rect.left, rect.top]]
                draw.line(rect_points, (128, 128, 128))

            # look for any tracks that occur on this frame
            for id, track in enumerate(tracker.tracks):

                frame_offset = frame_number - track.start_frame
                if frame_offset >= 0 and frame_offset < len(track.bounds_history)-1:
                    # display the track
                    rect = track.bounds_history[frame_offset]

                    # stub: make sure frame numbers are correct
                    if frame_number != rect.frame_index:
                        print("[{}] desync, {} {} {}".format(id, frame_number, rect.frame_index, track.start_frame))

                    rect_points = [int(p * FRAME_SCALE) for p in [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom, rect.left, rect.top]]
                    draw.line(rect_points, track_colors[id % len(track_colors)])


            video_frames.append(np.asarray(img))

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break

        tools.write_mpeg(filename, video_frames)

    def run_tests(self, source_folder, tests_file):
        """ Processes file in test file and compares results to expected output. """

        # disable hints for tests
        self.hints = []

        tests = []
        test = None

        # we need to make sure all tests are redone every time.
        self.overwrite_mode = CPTVTrackExtractor.OM_ALL

        # load in the test data
        for line in open(tests_file, 'r'):
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
                continue

            if line.split()[0].lower() == 'track':
                if test == None:
                    raise Exception("Can not have track before source file.")
                expected_length, expected_movement = [int(x) for x in line.split()[1:]]
                test.tracks.append((expected_length, expected_movement))
            else:
                test = TrackerTestCase()
                test.source = line
                tests.append(test)

        print("Found {0} test cases".format(len(tests)))

        for test in tests:
            self.run_test(source_folder, test)


def parse_params():

    parser = argparse.ArgumentParser()

    parser.add_argument('target', default='all', help='Target to process, "all" processes all folders, "test" runs test cases, or a "cptv" file to run a single source.')

    parser.add_argument('-o', '--output-folder', default=os.path.join(DEFAULT_BASE_PATH,"tracks"), help='Folder to output tracks to')
    parser.add_argument('-s', '--source-folder', default=os.path.join(DEFAULT_BASE_PATH,"clips"), help='Source folder root with class folders containing CPTV files')
    parser.add_argument('-c', '--color-map', default="custom_colormap.dat", help='Colormap to use when exporting MPEG files')
    parser.add_argument('-p', '--enable-previews', action='count', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-t', '--test-file', default='tests.txt', help='File containing test cases to run')
    parser.add_argument('--high-quality-optical-flow', action='store_true', default=False, help='Enables high quality optical flow (much slower).')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')
    parser.add_argument('-w', '--workers', default='0', help='Number of worker threads to use.  0 disables worker pool and forces a single thread.')
    parser.add_argument('-f', '--force-overwrite', default='old', help='Overwrite mode.  Options are all, old, or none.')
    parser.add_argument('-i', '--show-build-information', action='count', help='Show openCV build information and exit.')
    parser.add_argument('--disable-track-filters', action='count', help='Disables filtering of poor quality tracks.')

    args = parser.parse_args()

    if args.show_build_information:
        print(cv2.getBuildInformation())
        return

    # setup extractor
    extractor = CPTVTrackExtractor(args.output_folder)

    extractor.workers_threads = int(args.workers)
    if extractor.workers_threads >= 1:
        print("Using {0} worker threads".format(extractor.workers_threads))

    # set overwrite mode
    if args.force_overwrite.lower() not in ['all','old','none']:
        raise Exception("Valid overwrite modes are all, old, or none.")
    extractor.overwrite_mode = args.force_overwrite.lower()

    # set optical flow
    extractor.reduced_quality_optical_flow = not args.high_quality_optical_flow

    # set verbose
    extractor.verbose = args.verbose

    # this colormap is specially designed for heat maps
    extractor.colormap = tools.load_colormap(args.color_map)

    # load hints.  Hints are a way to give extra information to the tracker when necessary.
    extractor.load_hints("hints.txt")

    extractor.enable_previews = args.enable_previews

    extractor.source_folder = args.source_folder
    extractor.output_folder = args.output_folder

    # allow everything through
    extractor.disable_track_filters = args.disable_track_filters

    if extractor.enable_previews:
        print("Previews enabled.")

    if os.path.splitext(args.target)[1].lower() == '.cptv':
        # run single source
        source_file = tools.find_file(args.source_folder, args.target)
        tag = os.path.basename(os.path.dirname(source_file))
        extractor.overwrite_mode = CPTVTrackExtractor.OM_ALL
        extractor.process_file(source_file, tag=tag)
        return

    if args.target.lower() == 'test':
        print("Running test suite")
        extractor.run_tests(args.source_folder, args.test_file)
        return

    print('Processing tag "{0}"'.format(args.target))

    if args.target.lower() == 'all':
        extractor.process_all(args.source_folder)
        return
    else:
        extractor.process_folder(os.path.join(args.source_folder, args.target), tag=args.target, worker_pool_args=(trackdatabase.hdf5_lock,))
        return

def print_opencl_info():
    """ Print information about opencv support for opencl. """
    if cv2.ocl.haveOpenCL():
        if cv2.ocl.useOpenCL():
            print("OpenCL found and enabled, threads={}".format(cv2.getNumThreads()))
        else:
            print("OpenCL found but disabled")

def main():
    parse_params()

if __name__ == '__main__':

    # opencv sometimes uses too many threads which can reduce performance.  We are running a worker pool which makes
    # better use of multiple cores, so best to leave thread count per process reasonably low.
    # there is quite a big difference between 1 thread and 2, but after that gains are very minimal, and a lot of
    # cpu time gets wasted, starving the other workers.
    cv2.setNumThreads(2)

    print_opencl_info()
    main()

