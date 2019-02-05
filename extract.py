"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import cv2

from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools
from ml_tools.config import Config

from track.cptvtrackextractor import CPTVTrackExtractor

import os

import argparse

__version__ = '1.1.0'

def parse_params():

    parser = argparse.ArgumentParser()

    parser.add_argument('target', default='all', help='Target to process, "all" processes all folders, "test" runs test cases, "clean" to remove banned clips from db, or a "cptv" file to run a single source.')

    parser.add_argument('-p', '--show-previews', action='count', help='Show previews for tracks (can be slow)')
    parser.add_argument('-t', '--test-file', default='tests.txt', help='File containing test cases to run')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')
    parser.add_argument('-f', '--force-overwrite', default='old', help='Overwrite mode.  Options are all, old, or none.')
    parser.add_argument('-i', '--show-build-information', action='count', help='Show openCV build information and exit.')

    config = Config.load()

    args = parser.parse_args()

    if args.show_build_information:
        print(cv2.getBuildInformation())
        return

    # setup extractor
    extractor = CPTVTrackExtractor(config)

    # override previews if thrue
    if args.show_previews:
        extractor.preview_tracks = True

    # override verbose if true
    if args.verbose:
        extractor.verbose = True

    # # allow everything through
    # extractor.disable_track_filters = args.disable_track_filters

    if extractor.enable_previews:
        print("Previews enabled.")

    if extractor.verbose:
        print("Verbose description enabled.")


    if os.path.splitext(args.target)[1].lower() == '.cptv':
        # run single source
        source_file = tools.find_file(config.source_folder, args.target)
        if source_file is None:
            if not os.path.isfile(args.target):
                print("Could not locate file '" + args.target + "'")
                return
            source_file = args.target
        print("Processing file '" + source_file + "'")
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
        extractor.clean_all()
        extractor.process_all(config.source_folder)
        return
    if args.target.lower() == 'clean':
        extractor.clean_all()
        return
    else:
        extractor.process_folder(os.path.join(config.source_folder, args.target), tag=args.target, worker_pool_args=(trackdatabase.hdf5_lock,))
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

