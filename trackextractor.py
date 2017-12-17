"""
Module to handle tracking of objects in thermal video.
"""

# we need to use a non GUI backend.  AGG works but is quite slow so I used SVG instead.
import matplotlib
matplotlib.use("SVG")

import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.ndimage
import cv2

from cptv import CPTVReader

import pytz
import datetime
import dateutil
import time

import os
import json
import pickle

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, TrackingFrame):
            return (int(obj.left), int(obj.top), int(obj.right), int(obj.bottom))
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def get_image_subsection(image, bounds, window_size, boundary_value=None):
    """
    Returns a subsection of the original image bounded by bounds.
    Area outside of frame will be filled with boundary_value.  If None the median value will be used.
    """

    # todo: rewrite. just use the opencv / numpy padding function...

    # cropping method.  just center on the bounds center and take a section there.

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]

    # for some reason I write this to only work with even window sizes?
    window_half_width, window_half_height = window_size[0] // 2, window_size[1] // 2
    window_size = (window_half_width * 2, window_half_height * 2)
    image_height, image_width, channels = image.shape

    # find how many pixels we need to pad by
    padding = (max(window_size)//2)+1

    midx = int(bounds.mid_x + padding)
    midy = int(bounds.mid_y + padding)

    if boundary_value is None: boundary_value = np.median(image)

    # note, we take the median of all channels, should really be on a per channel basis.
    enlarged_frame = np.ones([image_height + padding*2, image_width + padding*2, channels], dtype=np.float16) * boundary_value
    enlarged_frame[padding:-padding,padding:-padding] = image

    sub_section = enlarged_frame[midy-window_half_width:midy+window_half_width, midx-window_half_height:midx+window_half_height]

    width, height, channels = sub_section.shape
    if int(width) != window_size[0] or int(height) != window_size[1]:
        print("Warning: subsection wrong size. Expected {} but found {}".format(window_size,(width, height)))

    if channels == 1:
        sub_section = sub_section[:,:,0]

    return sub_section

def load_tracker_stats(filename):
    """
    Loads a stats file for a processed clip.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """

    with open(filename, 'r') as t:
        # add in some metadata stats
        stats = json.load(t)

    stats['date_time'] = dateutil.parser.parse(stats['date_time'])
    return stats


def load_track_stats(filename):
    """
    Loads a stats file for a processed track.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """

    with open(filename, 'r') as t:
        # add in some metadata stats
        stats = json.load(t)

    stats['timestamp'] = dateutil.parser.parse(stats['timestamp'])
    return stats


def apply_threshold(frame, threshold = 50.0):
    """ Creates a binary mask out of an image by applying a threshold.
        Any pixels more than the threshold are set 1, all others are set to 0.
        A blur is also applied as a filtering step
    """
    cv2.setNumThreads(2)
    thresh = cv2.GaussianBlur(frame.astype(np.float32), (5,5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh


def normalise(x):
    x = x.astype(np.float32)
    return (x - np.mean(x)) / max(0.000001, float(np.std(x)))

class TrackingFrame:
    """ Defines a rectangle by the topleft point and width / height. """
    def __init__(self, topleft_x, topleft_y, width, height, mass = 0, id = 0):
        """ Defines new rectangle. """
        self.x = topleft_x
        self.y = topleft_y
        self.width = width
        self.height = height
        self.mass = mass
        self.id = id

    def copy(self):
        return TrackingFrame(self.x, self.y, self.width, self.height, self.mass, self.id)

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
        return "({0},{1},{2},{3})".format(self.left, self.top, self.right, self.bottom)

    def __str__(self):
        return "<({0},{1})-{2}x{3}>".format(self.x, self.y, self.width, self.height)


class Track:
    """ Defines an object tracked through the video frames."""

    """ keeps track of which id number we are up to."""
    _track_id = 1

    def __init__(self, x, y, width, height, mass = 0):

        self.bounds = TrackingFrame(x, y, width, height)
        self.origin = (self.bounds.mid_x, self.bounds.mid_y)
        self.first_frame = 0

        self.tracker = None

        # counts number of frames since we last saw target.
        self.frames_since_target_seen = 0

        self.id = Track._track_id
        Track._track_id += 1

        self.vx = 0.0
        self.vy = 0.0

        self.mass = mass

        # history for each frame in track
        self.mass_history = []

        self.bounds_history = [self.bounds.copy()]

        # used to record prediction of what kind of animal we are tracking
        self.prediction_history = []

        # average mass
        self.average_mass = 0.0
        # how much this track has moved
        self.movement = 0.0
        # how many pixels we have moved from origin
        self.max_offset = 0.0
        # how likely this is to be a valid track
        self.score = 0.0

        # stores various stats about this track.
        self.stats = {}

        # starting time of track
        self.start_time = None

    def __repr__(self):
        return "({0},{1})".format(self.bounds.x, self.bounds.y)

    @property
    def offsetx(self):
        """ Offset from where object was originally detected. """
        return self.bounds.mid_x - self.origin[0]

    @property
    def offsety(self):
        """ Offset from where object was originally detected. """
        return self.bounds.mid_x - self.origin[0]

    @property
    def frames(self):
        """ Number of frames this track has history for. """
        return len(self.bounds_history)

    @property
    def end_time(self):
        return self.start_time + datetime.timedelta(seconds=self.frames / 9.0)

    def get_frame(self, frame_number):
        """
        Gets 64x64 frame for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown
        :param frame_number: the frame number where 0 is the first frame of track.
        :return: numpy array of size [64,64,5] where channels are thermal, filtered, u, v, mask
        """

        if self.tracker is None:
            raise Exception("Tracker must be assigned to track before frames can be fetched.")

        if frame_number < 0 or frame_number >= self.frames:
            raise Exception("Frame {} is out of bounds for track with {} frames".format(
                frame_number, self.frames))

        bounds = self.bounds_history[frame_number]
        tracker_frame = self.first_frame + frame_number

        # window size must be even for get_image_subsection to work.
        window_size = (max(TrackExtractor.WINDOW_SIZE, bounds.width, bounds.height) // 2) * 2

        thermal = get_image_subsection(self.tracker.frames[tracker_frame], bounds, (window_size, window_size))
        filtered = get_image_subsection(self.tracker.filtered_frames[tracker_frame], bounds, (window_size, window_size), 0)
        flow = get_image_subsection(self.tracker.flow_frames[tracker_frame], bounds, (window_size, window_size), 0)
        mask = get_image_subsection(self.tracker.mask_frames[tracker_frame], bounds, (window_size, window_size), 0)

        if window_size != TrackExtractor.WINDOW_SIZE:
            scale = TrackExtractor.WINDOW_SIZE / window_size
            thermal = scipy.ndimage.zoom(np.float32(thermal), (scale, scale), order=1)
            filtered = scipy.ndimage.zoom(np.float32(filtered), (scale, scale), order=1)
            flow = scipy.ndimage.zoom(np.float32(flow), (scale, scale, 1), order=1)
            mask = scipy.ndimage.zoom(np.float32(mask), (scale, scale), order=1)

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        frame = np.float16(np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=2))

        return frame


    def get_velocity_from_flow(self, flow, mask):
        """ sets velocity from flow """
        track_flow = get_image_subsection(flow, self.bounds, (self.bounds.width, self.bounds.height), 0)
        track_mask = get_image_subsection(mask, self.bounds, (self.bounds.width, self.bounds.height), 0)

        # make sure we are the one on the mask
        track_mask[track_mask != self.id] = 0
        track_mask[track_mask > 0] = 1

        # too few pixels to work with.
        if np.sum(track_mask) < 2:
            self.vx = self.vy = 0.0
            return

        # we average the velocity of every pixel in our mask, but ignore the others.
        track_flow = track_flow[:, :] * track_mask[:, :, np.newaxis]
        self.vx = np.sum(track_flow[:, :, 0]) / np.sum(track_mask)
        self.vy = np.sum(track_flow[:, :, 1]) / np.sum(track_mask)

    def get_track_region_score(self, region):
        """
        Calculates a score between this track and a region of interest.  Regions that are close the the expected
        location for this track are given high scores, as are regions of a similar size.
        """
        expected_x = int(self.bounds.mid_x + self.vx)
        expected_y = int(self.bounds.mid_y + self.vy)

        distance = ((region.mid_x - expected_x) ** 2 + (region.mid_y - expected_y) ** 2) ** 0.5

        # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
        # area is padded with 50 pixels so small regions don't change too much
        size_difference = (abs(region.area - self.bounds.area) / (self.bounds.area+50)) * 100

        return distance, size_difference

    def sync_new_location(self, region):
        """ Work out out estimated new location for the frame using last position
            and movement vectors as an initial guess. """

        # reset our counter
        self.frames_since_target_seen = 0

        # just follow target.
        old_x, old_y = self.bounds.mid_x, self.bounds.mid_y
        self.bounds.x = region.x
        self.bounds.y = region.y
        self.bounds.width = region.width
        self.bounds.height = region.height
        self.mass = region.mass
        self.bounds.id = region.id

        self.vx = self.bounds.mid_x - old_x
        self.vy = self.bounds.mid_y - old_y

class TrackExtractor:
    """ Tracks objects within a CPTV thermal video file. """

    # these should really be in some kind of config file...

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
    MAX_MEAN_TEMPERATURE_THRESHOLD = 3800

    # any clips with a temperature dynamic range greater than this will be excluded
    MAX_TEMPERATURE_RANGE_THRESHOLD = 2000

    # if the mean pixel change is below this threshold then classify the video as having a static background
    STATIC_BACKGROUND_THRESHOLD = 5.0

    def __init__(self, full_path, max_frames = None):
        """
        Create a Tracker object
        :param full_path: path and filename of CPTV file to process
        """

        start = time.time()

        # date and time when video starts
        self.video_start_time = None

        self.frames = []
        self.track_history = {}
        self.load(open(full_path, 'rb'), max_frames)
        self.tag = "UNKNOWN"
        self.source = os.path.split(full_path)[1]
        self.tracks = []

        self.regions = []
        self.mask_frames = []
        self.filtered_frames = []
        self.flow_frames = []
        self.delta_frames = []

        # faster, but less accurate optical flow.  About 3 times faster.
        self.reduced_quality_optical_flow = False

        # default colormap to use for outputting preview files.
        self.colormap = plt.cm.jet

        self.verbose = False

        # if true excludes any videos with backgrounds that change too much.
        self.exclude_non_static_videos = True

        # the classifer to use to classify tracks
        self.classifier = None

        # find background
        self.background, self.auto_threshold = self.get_background()
        self.average_background_delta = self.get_background_average_change()
        self.is_static_background = self.average_background_delta < TrackExtractor.STATIC_BACKGROUND_THRESHOLD

        # If set to a number only this many frames will be used.
        self.max_tracks = None
        self.stats = self._get_clip_stats()

        self.stats['time_per_frame'] = {}
        self.stats['time_per_frame']['load'] = (time.time() - start) * 1000 / len(self.frames)


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

        # calculate the total time
        time_stats = self.stats['time_per_frame']
        time_stats['total'] = \
            time_stats.get('load',0.0) + \
            time_stats.get('extract', 0.0) + \
            time_stats.get('optical_flow', 0.0) + \
            time_stats.get('export', 0.0) + \
            time_stats.get('preview', 0.0)

        # force time per frame to rounded numbers
        for k,v in time_stats.items():
            time_stats[k] = int(v)

            # we need to convert datetime to a string so it will serialise through json
        with open(filename, 'w') as stats_file:
            json.dump(self.stats, stats_file, indent=4,  cls=CustomJSONEncoder)


    def load(self, source, max_frames = None):
        """
        Load frames from a CPTV file.
        :param source: source file
        :param max_frames: maximum number of frames to load, None will load the whole video (default)
        """
        reader = CPTVReader(source)
        self.frames = []
        for i, (frame, offset) in enumerate(reader):
            self.frames.append(frame.copy())
            if max_frames is not None and i >= max_frames:
                break

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
            rect = TrackingFrame(stats[i, 0] - padding, stats[i, 1] - padding, stats[i, 2] + padding * 2,
                                 stats[i, 3] + padding * 2, stats[i,4], i)
            rects.append(rect)

        return (rects, markers) if include_markers else rects

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
        threshold = np.percentile(deltas, q=TrackExtractor.THRESHOLD_PERCENTILE) / 2

        # cap the threshold to something reasonable
        if threshold < 10.0:
            threshold = 10.0
        if threshold > 50.0:
            threshold = 50.0

        return background, float(threshold)

    def extract(self):
        """
        Extract regions of interest from frames, and create some initial tracks.
        """

        # fisrt call to opencv is really slow, and throws out the timings, so I run cv2 command here
        # to warm up the library.
        x = cv2.medianBlur(np.zeros((32,32), dtype=np.float32), 5)

        if self.exclude_non_static_videos and not self.is_static_background:
            return

        start = time.time()
        optical_flow_time = 0.0

        if self.is_static_background:
            mask, threshold = self.background, self.auto_threshold
        else:
            # just use a blank mask
            mask = np.zeros_like(self.frames[0])
            threshold = 50.0

        active_tracks = []

        # reset frame history
        self.regions = []
        self.mask_frames = []
        self.filtered_frames = []
        self.flow_frames = []
        self.delta_frames = []

        # don't process clips that are too hot.
        if self.stats['mean_temp'] > TrackExtractor.MAX_MEAN_TEMPERATURE_THRESHOLD:
            return

        # don't process clips with too hot a temperature difference
        if self.stats['max_temp'] - self.stats['min_temp'] > TrackExtractor.MAX_TEMPERATURE_RANGE_THRESHOLD:
            return

        Track._track_id = 1

        tvl1 = cv2.createOptFlow_DualTVL1()
        if self.reduced_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            tvl1.setTau(1/4)
            tvl1.setScalesNumber(3)
            tvl1.setWarpingsNumber(3)
            tvl1.setScaleStep(0.5)

        # calculate filtered frames
        for frame in self.frames:
            # create a filtered frame
            filtered = frame - mask
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0
            self.filtered_frames.append(filtered)

        for frame_number in range(len(self.frames)):

            frame = self.frames[frame_number]

            # find regions of interest in this frame
            new_regions, markers = self._get_regions_of_interest(frame - mask, threshold, include_markers=True)

            self.mask_frames.append(markers)

            # calculate optical flow.
            flow_start_time = time.time()
            flow = np.zeros([frame.shape[0], frame.shape[1], 2], dtype=np.uint8)
            if frame_number > 0:
                # divide by two so we don't clip too much with hotter targets.
                current_gray_frame = (self.filtered_frames[frame_number-1] / 2).astype(np.uint8)
                next_gray_frame = (self.filtered_frames[frame_number] / 2).astype(np.uint8)

                # the tvl1 algorithm will take is many threads as it can.  On machines with many cores this ends up
                # being very inefficent.  For example this takes 80ms on 1 thread, 60ms on 2, and 50ms on 4, so the
                # gains are very deminising.  However the cpu will be pegged at full.  A better strategy is to simply
                # run additional instances of the Tracker in parallel
                cv2.setNumThreads(2)
                flow = tvl1.calc(current_gray_frame, next_gray_frame, flow)

            optical_flow_time += (time.time() - flow_start_time)

            flow = flow.astype(np.float32)
            self.flow_frames.append(flow)

            # find a delta frame
            if frame_number > 0:
                self.delta_frames.append(self.filtered_frames[frame_number].astype(np.float32) - self.filtered_frames[frame_number-1])
            else:
                self.delta_frames.append(frame * 0.0)

            # work out the best matchings for tracks and regions of interest
            scores = []
            for track in active_tracks:
                for region in new_regions:
                    distance, size_change = track.get_track_region_score(region)

                    if distance > 30:
                        continue
                    if size_change > 100:
                        continue
                    scores.append((distance, track, region))

            # apply matchings in a greedly.  Low score is best.
            matched_tracks = set()
            used_regions = set()

            scores.sort(key=lambda record: record[0])
            results = []

            for (score, track, region) in scores:
                # don't match a track twice
                if track in matched_tracks or region in used_regions:
                    continue
                track.sync_new_location(region)
                used_regions.add(region)
                matched_tracks.add(track)
                results.append((track, score))

            # update bounds history
            for track in active_tracks:
                track.bounds_history.append(track.bounds.copy())

            # create new tracks for any unmatched regions
            for region in new_regions:

                if region in used_regions:
                    continue

                # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
                overlaps = [track.bounds.overlap_area(region) for track in active_tracks]
                if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                    continue

                track = Track(region.x, region.y, region.width, region.height, region.mass)
                track.start_time = self.video_start_time + datetime.timedelta(seconds=frame_number / 9.0)
                track.first_frame = frame_number
                track.tracker = self
                active_tracks.append(track)
                self.track_history[track] = []

            # check if any tracks did not find a matched region
            for track in [track for track in active_tracks if track not in matched_tracks]:
                # we lost this track.  start a count down, and if we don't get it back soon remove it
                track.frames_since_target_seen += 1

            # remove any tracks that have not seen their target in 9 frames
            active_tracks = [track for track in active_tracks if track.frames_since_target_seen < 9]

            self.regions.append(rect.copy() for rect in new_regions)

            # step 5. record history.
            for track in active_tracks:
                self.track_history[track].append(
                    (frame_number, track.bounds.copy(), (track.vx, track.vy), (track.offsetx, track.offsety), track.mass))

        self.tracks = self.track_history.keys()
        self.get_tracks_statistics()

        self.stats['time_per_frame']['optical_flow'] = optical_flow_time * 1000 / len(self.frames)
        self.stats['time_per_frame']['extract'] = ((time.time() - start) * 1000 / len(self.frames)) - self.stats['time_per_frame']['optical_flow']

        """ Removes tracks with too poor a score to be used. """
        if self.max_tracks is not None and self.max_tracks < len(self.tracks):
            print(" -using only {0} tracks out of {1}".format(self.max_tracks, len(self.tracks)))
            self.tracks = self.tracks[:self.max_tracks]

    def get_tracks_statistics(self):
        """ Record stats on each track, including assigning it a score.  Also sorts tracks by score and filters out
            poor tracks. """

        track_scores = []

        counter = 1
        for track in self.tracks:

            history = self.track_history[track]

            track_length = len(history)

            # calculate movement statistics
            track.movement = sum(
                (vx ** 2 + vy ** 2) ** 0.5 for (frame_number, bounds, (vx, vy), (dx, dy), mass) in history)
            track.max_offset = max(
                (dx ** 2 + dy ** 2) ** 0.5 for (frame_number, bounds, (vx, vy), (dx, dy), mass) in history)

            # find total per frame deltas in this region
            deltas = []
            for (frame_number, bounds, _, _, _) in history:
                deltas.append(get_image_subsection(self.delta_frames[frame_number], bounds, (TrackExtractor.WINDOW_SIZE, TrackExtractor.WINDOW_SIZE), 0))
            deltas = np.asarray(deltas, dtype = np.float32)
            track.delta_std = float(np.std(deltas))
            track.stats['delta'] = track.delta_std

            movement_points = (track.movement ** 0.5) + track.max_offset
            delta_points = track.delta_std * 25.0

            track.score = movement_points + delta_points

            track.mass_history = list([int(mass) for (frame_number, bounds, (vx, vy), (dx, dy), mass) in history])
            track.average_mass = np.mean(track.mass_history)

            track.duration = track_length / 9.0

            if self.verbose:
                print(" - track duration:{:.1f}sec offset:{:.1f}px delta:{:.1f} mass:{:.1f}px".format(
                    track_length, track.max_offset, track.delta_std, track.average_mass
                ))

            # discard any tracks that are less than 3 seconds long (27 frames)
            # these are probably glitches anyway, or don't contain enough information.
            if track_length < 9 * 3:
                continue

            # discard tracks that do not move enough
            if track.max_offset < 4.0:
                continue

            # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
            if track.delta_std < 1.0:
                continue

            # discard tracks that do not have enough enough average mass.
            if track.average_mass < 2.0:
                continue

            track_scores.append((track.score, track))

            counter += 1

        track_scores.sort(reverse=True)
        self.tracks = [track for (score, track) in track_scores]

    def export_tracks(self, filename):
        """
        Export tracks to given filename base.  An TRK file will be exported.
        :param filename: full path and filename base name to export track to (-[track_number]) will be appended
        """

        # todo: would be great to just have a proper segment class that handles most of the code in this function...

        start = time.time()

        base_filename = os.path.splitext(filename)[0]

        for counter, track in enumerate(self.tracks):

            history = self.track_history[track]

            TRK_filename = base_filename + "-" + str(counter+1) + ".trk"
            Stats_filename = base_filename + "-" + str(counter+1) + ".txt"

            # export frames
            window_frames = []
            filtered_frames = []
            mask_frames = []
            flow_frames = []
            motion_vectors = []

            draw_frames = []

            # export the track file
            for frame_number, bounds, (vx, vy), (dx, dy), mass in history:

                if (frame_number - track.first_frame) >= track.frames:
                    print("warning... track frame out of bounds", frame_number, track.first_frame, track.frames)
                    continue

                frame = track.get_frame(frame_number - track.first_frame)

                # cast appropriately
                window_frames.append(np.float16(frame[:,:,0]))
                filtered_frames.append(np.float16(frame[:,:,1]))
                flow_frames.append(np.float16(frame[:,:,2:3+1]))
                mask_frames.append(np.uint8(frame[:,:,4]))

                motion_vectors.append((vx, vy))

            # export track stats.
            save_file = {}
            save_file['track_id'] = track.id
            save_file['frames'] = window_frames
            save_file['filtered_frames'] = filtered_frames
            save_file['mask_frames'] = mask_frames
            save_file['flow_frames'] = flow_frames
            save_file['motion_vectors'] = motion_vectors
            save_file['background'] = self.background

            stats = {}
            stats['id'] = track.id
            stats['score'] = track.score
            stats['movement'] = track.movement
            stats['average_mass'] = track.average_mass
            stats['max_offset'] = track.max_offset
            stats['timestamp'] = self.video_start_time
            stats['duration'] = track.duration
            stats['tag'] = self.tag
            stats['origin'] = track.origin
            stats['filename'] = self.source
            stats['threshold'] = self.auto_threshold
            stats['confidence'] = self.stats.get('confidence',0)
            stats['trap'] = self.stats.get('trap','')
            stats['event'] = self.stats.get('event','')
            stats['is_static_background'] = self.is_static_background

            # add in any stats generated during analysis.
            stats.update(track.stats)

            stats['mass_history'] = track.mass_history
            stats['bounds_history'] = track.bounds_history

            if len(track.mass_history) != len(window_frames):
                print("mass history mismatch", len(track.mass_history), len(window_frames))

            # save out track data
            pickle.dump(save_file, open(TRK_filename, 'wb'))

            with open(Stats_filename, 'w') as f:
                json.dump(stats, f, indent=4, cls=CustomJSONEncoder)

        self.stats['time_per_frame']['export'] = (time.time() - start) * 1000 / len(self.frames)
