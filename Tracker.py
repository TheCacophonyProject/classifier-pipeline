"""
Module to handle tracking of objects in thermal video.
"""

# we need to use a non GUI backend.  AGG works but is quite slow so I used SVG instead.
import matplotlib
matplotlib.use("SVG")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
import numpy as np
from cptv import CPTVReader
import cv2
import os
import pytz
import json
import datetime
import matplotlib.animation as manimation


def load_tracker_stats(filename):
    """
    Loads a stats file for a processed clip.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """

    with open(filename, 'r') as stats_file:
        # add in some metadata stats
        stats = json.load(stats_file.read())

    stats['date_time'] = datetime.dateutil.parse(stats['date_time'])
    return stats


def apply_threshold(frame, threshold = 'auto'):
    """ Creates a binary mask out of an image by applying a threshold.
        If threshold is not set a blend of the median, and max value of the image will be used.
    """
    if threshold == 'auto': threshold = (np.median(frame) + (25.0 if Tracker.USE_BACKGROUND_SUBTRACTION else 50.0))
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    thresh = frame - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh


def get_image_subsection(image, bounds, window_size, boundary_value = None):
    """
    Returns a subsection of the original image bounded by bounds.
    Area outside of frame will be filled with boundary_vaule.  If None the median value will be used.
    """

    # cropping method.  just center on the bounds center and take a section there.

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]

    padding = 50

    midx = int(bounds.mid_x + padding)
    midy = int(bounds.mid_y + padding)

    window_half_width, window_half_height = window_size[0] // 2, window_size[1] // 2

    image_height, image_width, channels = image.shape

    if boundary_value is None: boundary_value = np.median(image)

    # note, we take the median of all channels, should really be on a per channel basis.
    enlarged_frame = np.ones([image_height + padding*2, image_width + padding*2, channels]) * boundary_value
    enlarged_frame[padding:-padding,padding:-padding] = image

    sub_section = enlarged_frame[midy-window_half_width:midy+window_half_width, midx-window_half_width:midx+window_half_width]

    if channels == 1:
        sub_section = sub_section[:,:,0]

    return sub_section


class Rectangle:
    """ Defines a rectangle by the topleft point and width / height. """
    def __init__(self, topleft_x, topleft_y, width, height):
        """ Defines new rectangle. """
        self.x = topleft_x
        self.y = topleft_y
        self.width = width
        self.height = height

    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    @property
    def mid_x(self):
        return self.x + self.width / 2

    @property
    def mid_y(self):
        return self.y + self.height / 2

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def overlap_area(self, other):
        """ Compute the area overlap between this rectangle and another. """
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return x_overlap * y_overlap

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return "<({0},{1})-{2}x{3}>".format(self.x, self.y, self.width, self.height)


class TrackedObject:
    """ Defines an object tracked through the video frames."""

    """ keeps track of which id number we are up to."""
    _track_id = 0

    """ There is no target. """
    TARGET_NONE = 'none'

    """ Target has been acquired."""
    TARGET_ACQUIRED = 'acquired'

    """ Target has been lost."""
    TARGET_LOST = 'lost'

    """ Target collided with another target, or split."""
    TARGET_SPLIT = 'split'

    def __init__(self, x, y, width, height):

        self.bounds = Rectangle(x, y, width, height)
        self.status = TrackedObject.TARGET_NONE
        self.origion = (self.bounds.mid_x, self.bounds.mid_y)

        self.id = TrackedObject._track_id
        TrackedObject._track_id += 1

        self.vx = 0.0
        self.vy = 0.0

    def __repr__(self):
        return "({0},{1})".format(self.bounds.x, self.bounds.y)

    @property
    def offsetx(self):
        """ Offset from where object was originally detected. """
        return self.bounds.mid_x - self.origion[0]

    @property
    def offsety(self):
        """ Offset from where object was originally detected. """
        return self.bounds.mid_x - self.origion[0]

    def sync_new_location(self, regions_of_interest):
        """ Work out out estimated new location for the frame using last position
            and movement vectors as an initial guess. """
        gx = int(self.bounds.x + self.vx)
        gy = int(self.bounds.y + self.vy)

        new_bounds = Rectangle(gx, gy, self.bounds.width, self.bounds.height)

        # look for regions and calculate their overlap
        similar_regions = []
        overlapping_regions = []
        for region in regions_of_interest:
            overlap_fraction = (new_bounds.overlap_area(region) * 2) / (new_bounds.area + region.area)
            realtive_area_difference = abs(region.area - new_bounds.area) / new_bounds.area
            if overlap_fraction > 0.10 and realtive_area_difference < 0.5:
                similar_regions.append(region)
            if overlap_fraction > 0.10:
                overlapping_regions.append(region)

        if len(similar_regions) == 0:
            # lost target!
            self.status = TrackedObject.TARGET_LOST
            # print('lost')
        elif len(similar_regions) >= 2:
            # target split
            self.status = TrackedObject.TARGET_SPLIT
            # print('split', similar_regions)
        else:
            # just follow target.
            old_x, old_y = self.bounds.mid_x, self.bounds.mid_y
            self.bounds.x = similar_regions[0].x
            self.bounds.y = similar_regions[0].y
            self.bounds.width = similar_regions[0].width
            self.bounds.height = similar_regions[0].height
            # print("move to ",self.bounds.x, self.bounds.y)

            # work out out new velocity
            new_vx = self.bounds.mid_x - old_x
            new_vy = self.bounds.mid_y - old_y
            # smooth out the velocity changes a little bit.
            smooth = 0.9  # ema smooth
            self.vx = smooth * self.vx + (1 - smooth) * new_vx
            self.vy = smooth * self.vy + (1 - smooth) * new_vy

        return overlapping_regions


class Tracker:
    """ Tracks objects within a CPTV thermal video file. """

    # these should really be in some kind of config file...

    # used to keep track of version used to encode the tracks
    VERSION = 0.1

    # size of tracking window output in pixels.
    WINDOW_SIZE = 64

    # dpi to use for video, 100 is default, 50 is faster but hard to see tracking windows.
    VIDEO_DPI = 100

    # If enabled removes background by subtracting out the average pixels values before filtering.
    # Set to True to enable, False to disable, and 'auto' to enable only on stationary clips.

    USE_BACKGROUND_SUBTRACTION = 'auto'

    # auto threshold needs to find a near maximum value to calculate the threshold level
    # a better solution might be the mean of the max of each frame?
    THRESHOLD_PERCENTILE = 99.9

    # the coldest value to display when rendering previews
    TEMPERATURE_MIN = 2800
    TEMPERATURE_MAX = 4200

    # any clips with a mean temperature hotter than this will be excluded
    MAX_TEMPERATURE_THRESHOLD = 3800

    # if the mean pixel change is below this threshold then classify the video as having a static background
    STATIC_BACKGROUND_THRESHOLD = 5.0

    def __init__(self, full_path):
        """
        Create a Tracker object
        :param full_path: path and filename of CPTV file to process
        """

        self.frames = []
        self.track_history = {}
        self.load(open(full_path, 'rb'))
        self.tag = "UNKNOWN"
        self.source = os.path.split(full_path)[1]
        self.track_history = {}

        # find background
        self.background, self.auto_threshold = self.get_background()
        self.average_background_delta = self.get_background_average_change()
        self.is_static_background = self.average_background_delta < Tracker.STATIC_BACKGROUND_THRESHOLD

        # If set to a number only this many frames will be used.
        self.max_tracks = None
        self.stats = self._get_clip_stats()


    def _get_clip_stats(self):
        """
        Computes statitics for currently loaded clip and returns a dictionary containing the stats.
        :return: A dictionary containing stats from video clip.
        """
        result = {}
        local_tz = pytz.timezone('Pacific/Auckland')
        result['mean_temp'] = int(np.asarray(self.frames).mean())
        result['max_temp'] = int(np.asarray(self.frames).max())
        result['min_temp'] = int(np.asarray(self.frames).min())
        result['date_time'] = self.video_start_time.astimezone(local_tz)
        result['source'] = self.source
        result['is_static_background'] = self.is_static_background
        result['auto_threshold'] = self.auto_threshold
        result['is_night'] = self.video_start_time.astimezone(local_tz).time().hour >= 21 or self.video_start_time.astimezone(local_tz).time().hour <= 4
        result['average_background_delta'] = self.average_background_delta

        return result

    def print_stats(self):
        self.log_message(" - Temperature:{0} ({1}-{2}), Time of day: {3},Threshold: {4:.1f}".format(
            self.stats['mean_temp'], self.stats['min_temp'], self.stats['max_temp'],
            self.stats['time_of_day'].strftime("%H%M"), self.stats['auto_threshold']))

    def save_stats(self, filename):
        """ Writes stats to file. """

        # we need to convert datetime to a string so it will serialise through json
        stats= self.stats.copy()
        stats['date_time'] = stats['date_time'].isoformat()
        with open(filename, 'w') as stats_file:
            json.dump(stats, stats_file, indent=4)


    def load(self, source):
        """ Load frames from a CPTV file. """
        reader = CPTVReader(source)
        self.frames = [frame.copy() for (frame, offset) in reader]
        self.video_start_time = reader.timestamp


    def _get_regions_of_interest(self, frame, threshold, erosion=1, include_markers=False):
        """ Returns a list of bounded boxes for all regions of interest in the frame.
            Regions of interest are hotspots that stand out against the background.
        """

        thresh = np.asarray(apply_threshold(frame, threshold=(np.median(frame) + threshold)), dtype=np.uint8)

        # perform erosion
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=erosion)
        labels, markers, stats, centroids = cv2.connectedComponentsWithStats(eroded)

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = 12

        # find regions
        rects = []
        for i in range(1, labels):
            rect = Rectangle(stats[i, 0] - padding, stats[i, 1] - padding, stats[i, 2] + padding * 2,
                             stats[i, 3] + padding * 2)
            rects.append(rect)

        return (rects, markers) if include_markers else rects

    def _init_video(self, title, size):
        """
        Initialise an MPEG video with given title and size

        :param title: Title for the MPEG file
        :param size: tuple containing dims (width, height)
        :param colormap: colormap to use when outputting video

        :return: returns a tuple containing (figure, axis, image, and writer)
        """

        try:
            MPEGWriter = manimation.writers['ffmpeg']
        except Exception as e:
            raise Exception("MPEG Writer error {0}.  Try installing FFMPEG".format(e))

        metadata = dict(title=title, artist='Cacophony Project')
        writer = MPEGWriter(fps=9, metadata=metadata)

        # we create a figure of the appropriate dims.  Assuming 100 dpi
        figure_size = (size[0]/25, size[1]/25)

        fig, ax = plt.subplots(1, figsize = figure_size)
        data = np.zeros((size[1], size[0]),dtype=np.float32)
        ax.axis('off')

        im = plt.imshow(data, vmin=Tracker.TEMPERATURE_MIN , vmax=Tracker.TEMPERATURE_MAX)
        return (fig, ax, im, writer)


    def get_background_average_change(self):
        """
        Returns how much each pixel changes on average over the video.  Used to detect static backgrounds.
        :return: How much each pixel changes in value every frame.
        """
        delta = np.asarray(self.frames[1:],dtype=np.float32) - np.asarray(self.frames[:-1],dtype=np.float32)
        return float(np.mean(np.abs(delta)))


    def get_background(self):
        """
        Returns estimated background for video and threshold used.
        """

        background = np.percentile(np.asarray(self.frames), q=10.0, axis=0)

        deltas = np.reshape(self.frames - background, [-1])
        threshold = np.percentile(deltas, q=Tracker.THRESHOLD_PERCENTILE) / 2

        # cap the threshold to something reasonable
        if threshold < 10.0:
            threshold = 10.0
        if threshold > 50.0:
            threshold = 50.0

        return (background, float(threshold))

    def display(self, filename, colormap = None):
        """
        Exports tracking information to a video file for debugging.
        """

        if colormap is None: colormap = plt.cm.jet

        # setup the writer
        (fig, ax, im, writer) = self._init_video(filename, (160*2, 120*2))
        im.colormap = colormap

        # write video
        frame_number = 0
        with writer.saving(fig, filename, dpi=Tracker.VIDEO_DPI):
            for frame, marked, rects, flow, filtered in zip(self.frames, self.marked_frames, self.regions, self.flow_frames, self.filtered_frames):
                # marked is an image with each pixel's value being the label, 0...n for n objects
                # I multiply it here, but really I should use a seperate color map for this.
                # maybe I could multiply it modulo, and offset by some amount?

                # note: it would much be better to have 4 seperate figures, with their own colour
                # palettes, but this was easier to setup for the moment.
                filtered_frame = 1.5 * (frame - self.background) + Tracker.TEMPERATURE_MIN #use 28000 as baseline (black) background, but bump up the brightness a little.

                stacked = np.hstack((np.vstack((frame, marked*10000)),np.vstack((filtered_frame, self.background))))
                im.set_data(stacked)

                patch_list = []
                for rect in rects:
                    patch = patches.Rectangle((rect.x, rect.y), rect.width, rect.height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(patch)
                    patch_list.append(patch)

                fig.canvas.draw()
                writer.grab_frame()

                for patch in patch_list:
                    patch.remove()

                frame_number += 1

        plt.close(fig)


    def extract(self):
        """
        Extract regions of interest from frames.
        """

        if Tracker.USE_BACKGROUND_SUBTRACTION.lower() == 'auto':
            use_background_subtraction = self.is_static_background
        else:
            use_background_subtraction = Tracker.USE_BACKGROUND_SUBTRACTION

        if use_background_subtraction:
            mask, threshold = self.background, self.auto_threshold
        else:
            # just use a blank mask
            mask = np.zeros_like(self.frames[0])
            threshold = 50.0

        active_tracks = []

        self.regions = []
        self.marked_frames = []
        self.filtered_frames = []
        self.flow_frames = []

        # don't process clips that are too hot.
        if self.stats['mean_temp'] > Tracker.MAX_TEMPERATURE_THRESHOLD:
            return

        TrackedObject._track_id = 0

        prev_frame = self.frames[0]

        for frame_number, frame in enumerate(self.frames):

            # calculate optical flow
            self.flow_frames.append(cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            prev_frame = frame

            # step 1. find regions of interest in this frame
            new_regions, markers = self._get_regions_of_interest(frame - mask, threshold, include_markers=True)

            self.marked_frames.append(markers)

            filtered = frame-mask - threshold
            filtered[filtered < 0] = 0
            self.filtered_frames.append(filtered)

            used_regions = []

            # step 2. match these with tracked objects
            for track in active_tracks:
                # update each track.
                used_regions = used_regions + track.sync_new_location(new_regions)

            # step 3. create new tracks for any unmatched regions
            for region in new_regions:
                if region in used_regions:
                    continue
                #print("discovered new object:", region)
                active_tracks.append(TrackedObject(region.x, region.y, region.width, region.height))
                self.track_history[active_tracks[-1].id] = []

            # step 4. delete lost tracks
            for track in active_tracks:
                if track.status == TrackedObject.TARGET_LOST:
                    #print("target lost.")
                    pass

            active_tracks = [track for track in active_tracks if track.status != TrackedObject.TARGET_LOST]

            self.regions.append([track.bounds.copy() for track in active_tracks])

            # step 5. record history.
            for track in active_tracks:
                self.track_history[track.id].append(
                    (frame_number, track.bounds.copy(), (track.vx, track.vy), (track.offsetx, track.offsety)))


    def export(self, filename):
        """ export tracks to given filename base.  An MPEG and TRK file will be exported. """

        track_scores = []

        # gather usable tracks
        counter = 1
        for track_id in self.track_history.keys():

            history = self.track_history[track_id]
            track_length = len(history)

            # calculate movement statistics
            track_movement = sum(
                (vx ** 2 + vy ** 2) ** 0.5 for (frame_number, bounds, (vx, vy), (dx, dy)) in history)
            track_max_offset = max(
                (dx ** 2 + dy ** 2) ** 0.5 for (frame_number, bounds, (vx, vy), (dx, dy)) in history)

            track_origin = (history[0][1].mid_x, history[0][1].mid_y)

            track_score = track_movement + track_max_offset

            # discard any tracks that are less than 3 seconds long (27 frames)
            # these are probably glitches anyway, or don't contain enough information.
            if track_length < 9*3:
                continue

            # discard tracks that do not move enough
            if track_max_offset < 4.0:
                continue

            track_scores.append((track_score, track_id))

            # display some debuging output.
            print(" -track {0}: length {1} frames, movement {2:.2f}, max_offset {3:.2f}, score {4:.2f}".format(counter, track_length,
                                                                                        track_movement,
                                                                                        track_max_offset,
                                                                                        track_score))

            counter += 1

        track_scores.sort(reverse=True)
        ordered_tracks = [track_id for (score, track_id) in track_scores]
        if self.max_tracks is not None:
            print(" -using only {0} tracks out of {1}".format(self.max_tracks, len(ordered_tracks)))
            ordered_tracks = ordered_tracks [:self.max_tracks]

        for counter, track_id in enumerate(ordered_tracks):

            history = self.track_history[track_id]

            MPEG_filename = filename + "-" + str(counter+1 ) + ".mp4"
            TRK_filename = filename + "-" + str(counter+1) + ".trk"

            # export frames
            window_frames = []
            filtered_frames = []
            flow_frames = []
            motion_vectors = []

            # setup MPEG writer
            (fig, ax, im, writer) = self._init_video(MPEG_filename, (Tracker.WINDOW_SIZE, Tracker.WINDOW_SIZE))

            # process the frames
            with writer.saving(fig, MPEG_filename, dpi=Tracker.VIDEO_DPI):
                for frame_number, bounds, (vx, vy), (dx, dy) in history:


                    window_frames.append(get_image_subsection(self.frames[frame_number], bounds, (Tracker.WINDOW_SIZE, Tracker.WINDOW_SIZE)))
                    filtered_frames.append(get_image_subsection(self.filtered_frames[frame_number], bounds, (Tracker.WINDOW_SIZE, Tracker.WINDOW_SIZE),0))
                    flow_frames.append(get_image_subsection(self.flow_frames[frame_number], bounds, (Tracker.WINDOW_SIZE, Tracker.WINDOW_SIZE,0)))

                    motion_vectors.append((vx, vy))

                    draw_frame = get_image_subsection(self.filtered_frames[frame_number], bounds, (Tracker.WINDOW_SIZE, Tracker.WINDOW_SIZE))
                    draw_frame = 5 * draw_frame + Tracker.TEMPERATURE_MIN

                    im.set_data(draw_frame)
                    fig.canvas.draw()
                    writer.grab_frame()

            save_file = {}
            save_file['track_id'] = track_id
            save_file['version'] = Tracker.VERSION

            save_file['frames'] = window_frames
            save_file['filtered_frames'] = filtered_frames
            save_file['flow_frames'] = flow_frames
            save_file['track_movement'] = track_movement

            save_file['motion_vectors'] = motion_vectors
            save_file['track_max_offset'] = track_max_offset
            save_file['track_timestamp'] = self.video_start_time
            save_file['track_tag'] = self.tag
            save_file['track_origin'] = track_origin
            save_file['source_filename'] = self.source
            save_file['threshold'] = self.auto_threshold

            # bring confidence accross
            save_file['original_confidence'] = self.stats['confidence'] if 'confidence' in self.stats else None

            pickle.dump(save_file, open(TRK_filename, 'wb'))

            plt.close(fig)
