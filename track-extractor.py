"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

# we need to use a non GUI backend.  AGG works but is quite slow so I used SVG instead.
import matplotlib
matplotlib.use("SVG")

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pickle
import os
import Tracker

class CPTVTrackExtractor:
    """ Handles extracting tracks from CPTV files. """
    def __init__(self, out_folder):
        self.hints = {}
        self.colormap = plt.cm.jet
        self.verbose = True
        self.out_folder = out_folder
        self._init_MpegWriter()

    def _init_MpegWriter(self):
        """ setup our MPEG4 writer.  Requires FFMPEG to be installed. """
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        try:
            self.MpegWriter = manimation.writers['ffmpeg']
        except:
            self.log_warning("FFMPEG is not installed.  MPEG output disabled")
            self.MpegWriter = None

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
            print("Processing folder {0}".format(folder))
            self.process_folder(folder)
        print("Done.")

    def process_folder(self, folder_path, tag = None):
        """ Extract tracks from all videos in given folder.
            All tracks will be tagged with 'tag', which defaults to the folder name if not specified."""

        if tag is None:
            tag = os.path.basename(folder_path).upper()

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path) and os.path.splitext(full_path )[1].lower() == '.cptv':
                self.process_file(full_path, tag)

    def log_message(self, message):
        """ Record message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.verbose: print(message)

    def log_warning(self, message):
        """ Record warning message in log.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        print("Warning:",message)

    def process_file(self, full_path, tag, overwrite=False):
        """
        Extract tracks from specific file, and assign given tag.
        :param full: path: path to CPTV file to be processed
        :param tag: the tag to assign all tracks from this CPTV files
        :param overwrite: if true destination file will be overwritten
        """

        filename = os.path.split(full_path)[1]
        destination_folder = os.path.join(self.out_folder, tag.lower())
        destination_file = os.path.join(destination_folder, filename)

        # read additional information from hints file
        if filename in self.hints:
            max_tracks = self.hints[filename]
            if max_tracks == 0:
                return
        else:
            max_tracks = None

        # make destination folder if required
        try:
            os.stat(destination_folder)
        except:
            self.log_message(" Making path " +destination_folder)
            os.mkdir(destination_folder)

        # handle overwritting
        if os.path.exists(destination_file):
            if overwrite:
                self.log_message("Overwritting {0} [{1}]".format(filename, tag))
            else:
                return
        else:
            self.log_message("Processing {0} [{1}]".format(filename, tag))

        tracker = Tracker.Tracker(full_path)
        tracker.max_tracks = max_tracks
        tracker.tag = tag

        # display some debuging information
        """
        mean_temp = int(np.asarray(sequence.frames).mean())
        max_temp = int(np.asarray(sequence.frames).max())
        min_temp = int(np.asarray(sequence.frames).min())
        local_tz = pytz.timezone('Pacific/Auckland')
        time_of_day = sequence.video_start_time.astimezone(local_tz).time()
        self.log_message(" - Temperature:{0} ({3}-{4}), Time of day: {1},Threshold: {2:.1f}".format(mean_temp,
                            time_of_day.strftime("%H%M"),threshold,min_temp,max_temp))
        """

        tracker.extract()

        mpeg_filename = os.path.splitext(filename)[0]+".mp4"

        tracker.export(os.path.join(self.out_folder, tag, filename))

        tracker.display(os.path.join(self.out_folder, tag.lower(), mpeg_filename), self.colormap)


def main():

    extractor = CPTVTrackExtractor('d:\cac\\tracks')

    # this colormap is specially designed for heat maps
    extractor.load_custom_colormap('custom_colormap.dat')

    # load hints.  Hints are a way to give extra information to the tracker when necessary.
    extractor.load_hints("hints.txt")

    extractor.process('d:\\cac\\out')

main()

