"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

# we need to use a non GUI backend.  AGG works but is quite slow so I used SVG instead.
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle
import os
import Tracker
import ast
import glob
import argparse
import signal
import time

from multiprocessing import Pool

# default base path to use if no source or destination folder are given.
DEFAULT_BASE_PATH = "c:\\cac"

EXCLUDED_FOLDERS = ['false-positive','insect','other','unidentified','cat','dog']

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

class CPTVTrackExtractor:
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

        self.hints = {}
        self.colormap = plt.cm.jet
        self.verbose = False
        self.out_folder = out_folder
        self.overwrite_mode = CPTVTrackExtractor.OM_OLD_VERSION
        self.enable_previews = False
        self.display_times = False

        self.MPEGWriter  = None

        # number of threads to use when processing files.
        self.workers_threads = 1

        self._init_MPEGWriter()

    def _init_MPEGWriter(self):
        """ setup our MPEG4 writer.  Requires FFMPEG to be installed. """
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        try:
            self.MPEGWriter = manimation.writers['ffmpeg']
            self.log_message("FFMPEG found.")
        except:
            self.log_warning("FFMPEG is not installed.  MPEG output disabled")
            self.MPEGWriter = None

    def load_custom_colormap(self, filename):
        """ Loads a custom colormap used for creating MPEG previews of tracks. """

        if not os.path.exists(filename):
            return

        self.colormap = pickle.load(open(filename, 'rb'))

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

    def process(self, root_folder):
        """ Process all files in root folder.  CPTV files are expected to be found in folders corresponding to their
            class name. """

        folders = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if
                   os.path.isdir(os.path.join(root_folder, f))]

        for folder in folders:
            if os.path.basename(folder).lower() in EXCLUDED_FOLDERS:
                continue
            print("Processing folder {0}".format(folder))
            self.process_folder(folder)
        print("Done.")

    def process_file(self, full_path, tag, create_preview_file=False):
        """
        Extract tracks from specific file, and assign given tag.
        :param full_path: path: path to CPTV file to be processed
        :param tag: the tag to assign all tracks from this CPTV files
        :param create_preview_file: if enabled creates an MPEG preview file showing the tracking working.  This
            process can be quite time consuming.
        :returns the tracker object
        """

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
        tracker = Tracker.Tracker(full_path)
        tracker.max_tracks = max_tracks
        tracker.tag = tag
        tracker.verbose = self.verbose >= 2

        # read metadata
        meta_data_filename = os.path.splitext(full_path)[0] + ".dat"
        if os.path.exists(meta_data_filename):

            meta_data = ast.literal_eval(open(meta_data_filename, 'r').read())

            tag_count = len(meta_data['Tags'])

            # we can only handle one tagged animal at a time here.
            if tag_count != 1:
                print(" - Warning, too many tags in cptv files, ignoring.")
                return

            tracker.stats['confidence'] = meta_data['Tags'][0].get('confidence',0.0)
            tracker.stats['trap'] = meta_data['Tags'][0].get('trap','none')
            tracker.stats['event'] = meta_data['Tags'][0].get('event','none')
            tracker.stats['cptv_metadata'] = meta_data['Tags'][0]

        else:
            self.log_warning(" - Warning: no metadata found for file.")

        # pass the mpeg writer to the tracker so that it can output video files
        tracker.MPEGWriter = self.MPEGWriter

        # save some additional stats
        tracker.stats['version'] = CPTVTrackExtractor.VERSION

        tracker.extract()

        tracker.export(os.path.join(self.out_folder, tag, cptv_filename), use_compression=False,
                       include_track_previews=create_preview_file and self.MPEGWriter is not None)

        if create_preview_file == 2:
            tracker.display(os.path.join(self.out_folder, tag.lower(), preview_filename), self.colormap)

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

    def process_job_list(self, jobs):
        """ Processes a list of jobs. """
        if self.workers_threads == 0:
            # just process the jobs in the main thread
            for job in jobs: process_job(job)
        else:
            # send the jobs to a worker pool
            pool = Pool(self.workers_threads)
            try:
                # see https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
                pool.map_async(process_job, jobs, chunksize=1).get(timeout=10 ** 6)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("KeyboardInterrupt, terminating.")
                pool.terminate()
                exit()
            else:
                pool.close()

    def process_folder(self, folder_path, tag = None):
        """ Extract tracks from all videos in given folder.
            All tracks will be tagged with 'tag', which defaults to the folder name if not specified."""

        if tag is None:
            tag = os.path.basename(folder_path).upper()

        jobs = []

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path) and os.path.splitext(full_path )[1].lower() == '.cptv':
                jobs.append((self, full_path, tag, self.enable_previews))

        self.process_job_list(jobs)

    def log_message(self, message):
        """ Record message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.verbose: print(message)

    def log_warning(self, message):
        """ Record warning message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        print("Warning:",message)


    def needs_processing(self, stats_filename):
        """
        Opens a stats file and checks if this clip needs to be processed.
        :param stats_filename: the full path and filename of the stats file for the clip in question.
        :return: returns true if file should be overwritten, false otherwise
        """

        # if no stats file exists we haven't processed file, so reprocess
        if not os.path.exists(stats_filename):
            return True

        # otherwise check what needs to be done.
        if self.overwrite_mode == CPTVTrackExtractor.OM_ALL:
            return True
        elif self.overwrite_mode == CPTVTrackExtractor.OM_NONE:
            return False

        # read in stats file.
        stats = Tracker.load_tracker_stats(stats_filename)
        try:
            stats = Tracker.load_tracker_stats(stats_filename)
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

        tracker = self.process_file(source_file, 'test', create_preview_file=self.enable_previews)

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
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')
    parser.add_argument('-w', '--workers', default='0', help='Number of worker threads to use.  0 disables worker pool and forces a single thread.')
    parser.add_argument('-f', '--force-overwrite', default='old', help='Overwrite mode.  Options are all, old, or none.')

    args = parser.parse_args()

    # setup extractor

    extractor = CPTVTrackExtractor(args.output_folder)

    extractor.workers_threads = int(args.workers)
    if extractor.workers_threads >= 1:
        print("Using {0} worker threads".format(extractor.workers_threads))

    # set overwrite mode
    if args.force_overwrite.lower() not in ['all','old','none']:
        raise Exception("Valid overwrite modes are all, old, or none.")
    extractor.overwrite_mode = args.force_overwrite.lower()

    # set verbose
    extractor.verbose = args.verbose

    # this colormap is specially designed for heat maps
    extractor.load_custom_colormap(args.color_map)

    # load hints.  Hints are a way to give extra information to the tracker when necessary.
    extractor.load_hints("hints.txt")

    extractor.enable_previews = args.enable_previews
    extractor.display_times = args.enable_previews

    if extractor.enable_previews:
        print("Previews enabled.")

    if os.path.splitext(args.target)[1].lower() == '.cptv':
        # run single source
        source_file = find_file(args.source_folder, args.target)
        tag = os.path.basename(os.path.dirname(source_file))
        extractor.overwrite_mode = CPTVTrackExtractor.OM_ALL
        extractor.process_file(source_file, tag, args.enable_previews)
        return

    if args.target.lower() == 'test':
        print("Running test suite")
        extractor.run_tests(args.source_folder, args.test_file)
        return

    print('Processing tag "{0}"'.format(args.target))

    if args.target.lower() == 'all':
        extractor.process(args.source_folder)
        return
    else:
        extractor.process_folder(os.path.join(args.source_folder, args.target), args.target)
        return


def main():
    parse_params()

if __name__ == '__main__':
    main()

