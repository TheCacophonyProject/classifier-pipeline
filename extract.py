"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import cv2
from PIL import Image, ImageDraw
import numpy as np

from ml_tools.trackextractor import TrackExtractor
from ml_tools.cptvfileprocessor import CPTVFileProcessor
from ml_tools import tools

import matplotlib.pyplot as plt
import pickle
import os

import ast
import glob
import argparse
import time

import multiprocessing

# default base path to use if no source or destination folder are given.
DEFAULT_BASE_PATH = "c:\\cac"

EXCLUDED_FOLDERS = ['false-positive','other','unidentified']

def purge(dir, pattern):
    for f in glob.glob(os.path.join(dir, pattern)):
        os.remove(os.path.join(dir, f))

def find_file(root, filename):
    """
    Finds a file in root folder, or any subfolders.
    :param root: root folder to search file
    :param filename: exact time of file to look for
    :return: returns full path to file or None if not found.
    """
    for root, dir, files in os.walk(root):
        if filename in files:
            return os.path.join(root, filename)
    return None

class TrackEntry:
    """ Database entry for a track """

    def __init__(self):
        pass

class TrackerTestCase():
    def __init__(self):
        self.source = None
        self.tracks = []

def process_job(job):
    """ Just a wrapper to pass tupple containing (extractor, *params) to the process_file method. """
    extractor = job[0]
    params = job[1:]
    extractor.process_file(*params)
    time.sleep(0.001) # apparently gives me a chance to catch the control-c

class CPTVTrackExtractor(CPTVFileProcessor):
    """
    Handles extracting tracks from CPTV files.
    Maintains a database recording which files have already been processed, and some statistics parameters used
    during processing.
    """

    # version number.  Recorded into stats file when a clip is processed.
    VERSION = 6

    # all files will be reprocessed
    OM_ALL = 'all'

    # any clips with a lower version than the current will be reprocessed
    OM_OLD_VERSION = 'old'

    # no clips will be overwritten
    OM_NONE = 'none'

    def __init__(self, out_folder):

        CPTVFileProcessor.__init__(self)

        self.hints = {}
        self.colormap = plt.cm.jet
        self.verbose = False
        self.out_folder = out_folder
        self.overwrite_mode = CPTVTrackExtractor.OM_OLD_VERSION
        self.enable_previews = False

        # normally poor quality tracks are filtered out, enabling this will let them through.
        self.disable_track_filters = False

        self.reduced_quality_optical_flow = False

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
                    self.process_folder(os.path.join(root,folder), tag=folder.lower())

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
            # some longer clips generate ~70 tracks (because of poor tracking mostly) so for the moment we limit these.
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
        purge(destination_folder, base_filename + "*.trk")
        purge(destination_folder, base_filename + "*.mp4")
        purge(destination_folder, base_filename + "*.txt")

        # load the track
        tracker = TrackExtractor(full_path)
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
            tracker.stats['cptv_metadata'] = meta_data
        else:
            self.log_warning(" - Warning: no metadata found for file.")
            return

        # save some additional stats
        tracker.stats['version'] = CPTVTrackExtractor.VERSION

        tracker.extract()

        tracker.export_tracks(os.path.join(self.out_folder, tag, cptv_filename))

        # write a preview
        if self.enable_previews:
            self.export_mpeg_preview(os.path.join(destination_folder, preview_filename), tracker)

        tracker.save_stats(stats_path_and_filename)

        time_stats = tracker.stats['time_per_frame']
        self.log_message("Tracks: {} {:.1f}sec - Times (per frame): [total:{}ms]  load:{}ms extract:{}ms optical flow:{}ms export:{}ms preview:{}ms".format(
            len(tracker.tracks),
            sum(track.duration for track in tracker.tracks),
            time_stats.get('total',0.0),
            time_stats.get('load',0.0),
            time_stats.get('extract',0.0),
            time_stats.get('optical_flow',0.0),
            time_stats.get('export',0.0),
            time_stats.get('preview', 0.0)
        ))

        return tracker

    def needs_processing(self, source_filename):
        """
        Opens a stats file and checks if this clip needs to be processed.
        :param stats_filename: the full path and filename of the stats file for the clip in question.
        :return: returns true if file should be overwritten, false otherwise
        """

        stats_filename = os.path.join(self.output_folder,os.path.splitext(os.path.basename(source_filename))[0])+".txt"

        # if no stats file exists we haven't processed file, so reprocess
        if not os.path.exists(stats_filename):
            return True

        # otherwise check what needs to be done.
        if self.overwrite_mode == CPTVTrackExtractor.OM_ALL:
            return True
        elif self.overwrite_mode == CPTVTrackExtractor.OM_NONE:
            return False

        # read in stats file.
        try:
            stats = tools.load_tracker_stats(stats_filename)
        except Exception as e:
            self.log_warning("Invalid stats file "+stats_filename+" error:"+str(e))
            return True

        if self.overwrite_mode == CPTVTrackExtractor.OM_OLD_VERSION:
            return stats['version'] < CPTVTrackExtractor.VERSION

        raise Exception("Invalid overwrite mode {0}".format(self.overwrite_mode))


    def run_test(self, source_folder, test: TrackerTestCase):
        """ Runs a specific test case. """

        def are_similar(value, expected, relative_error = 0.2, abs_error = 2.0):
            """ Checks of value is similar to expected value. An expected value of 0 will always return true. """
            if expected == 0:
                return True
            return ((abs(value - expected) / expected) <= relative_error) or (abs(value - expected) <= abs_error)

        # find the file.  We looking in all the tag folder to make life simpler when creating the test file.
        source_file = find_file(source_folder, test.source)

        if source_file is None:
            print("Could not find {0} in root folder {1}".format(test.source, source_folder))
            return

        tracker = self.process_file(source_file, tag='test')

        # read in stats files and see how we did
        if len(tracker.tracks) != len(test.tracks):
            print("[Fail] {0} Incorrect number of tracks, expected {1} found {2}".format(test.source, len(test.tracks), len(tracker.tracks)))
            return

        for test_result, (expected_duration, expected_movement) in zip(tracker.tracks, test.tracks):
            if not are_similar(test_result.duration, expected_duration) or not are_similar(test_result.max_offset, expected_movement):
                print("[Fail] {0} Track too dissimilar expected {1} but found {2}".format(
                    test.source,
                    (expected_duration, expected_movement),
                    (test_result.duration, test_result.max_offset)))
            else:
                print("[PASS] {0}".format(test.source))

    def export_mpeg_preview(self, filename, tracker: TrackExtractor):
        """
        Exports tracking information preview to MPEG file.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 3.0

        start = time.time()

        video_frames = []
        track_colors = [(255,0,0),(0,255,0),(255,255,0),(128,255,255)]

        # write video
        frame_number = 0

        for frame, marked, rects, flow, filtered in zip(tracker.frames, tracker.mask_frames, tracker.regions, tracker.flow_frames, tracker.filtered_frames):

            # marked is an image with each pixel's value being the label, 0...n for n objects
            # I multiply it here, but really I should use a seperate color map for this.
            # maybe I could multiply it modulo, and offset by some amount?

            # really should be using a pallete here, I multiply by 10000 to make sure the binary mask '1' values get set to the brightest color (which is about 4000)
            # here I map the flow magnitude [ranges in the single didgits) to a temperature in the display range.
            flow_magnitude = (flow[:,:,0]**2 + flow[:,:,1]**2) ** 0.5
            stacked = np.hstack((np.vstack((frame, marked*10000)),np.vstack((3 * filtered + tools.TEMPERATURE_MIN, 200 * flow_magnitude + tools.TEMPERATURE_MIN))))

            img = tools.convert_heat_to_img(stacked, self.colormap, tools.TEMPERATURE_MIN, tools.TEMPERATURE_MAX)
            img = img.resize((int(img.width * FRAME_SCALE), int(img.height * FRAME_SCALE)), Image.NEAREST)
            draw = ImageDraw.Draw(img)

            # look for any regions of interest that occur on this frame
            for rect in rects:
                rect_points = [int(p * FRAME_SCALE) for p in [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom, rect.left, rect.top]]
                draw.line(rect_points, (128, 128, 128))

            # look for any tracks that occur on this frame
            for id, track in enumerate(tracker.tracks):
                frame_offset = frame_number - track.first_frame
                if frame_offset > 0 and frame_offset < len(track.bounds_history)-1:

                    # display the track
                    rect = track.bounds_history[frame_offset]
                    rect_points = [int(p * FRAME_SCALE) for p in [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom, rect.left, rect.top]]
                    draw.line(rect_points, track_colors[id % len(track_colors)])

            video_frames.append(np.asarray(img))

            frame_number += 1

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break

        tools.write_mpeg(filename, video_frames)

        tracker.stats['time_per_frame']['preview'] = (time.time() - start) * 1000 / len(tracker.frames)


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
    print(extractor.reduced_quality_optical_flow)

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
        source_file = find_file(args.source_folder, args.target)
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
        extractor.process_folder(os.path.join(args.source_folder, args.target), tag=args.target)
        return

def print_opencl_info():
    """ Print information about opencv support for opencl. """
    if cv2.ocl.haveOpenCL():
        if cv2.ocl.useOpenCL():
            print("OpenCL found and enabled")
        else:
            print("OpenCL found but disabled")

def main():
    parse_params()

if __name__ == '__main__':
    print_opencl_info()
    main()

