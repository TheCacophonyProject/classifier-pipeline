from cptv import CPTVReader
import cv2
import numpy as np
from track.framebuffer import FrameBuffer

class Clip:
  	PREVIEW = "preview"

  	FRAMES_PER_SECOND = 9
    def __init__(self, trackconfig):

        self.config = trackconfig

        # start time of video
        self.video_start_time = None
        # name of source file
        self.source_file = None

        # per frame temperature statistics for thermal channel
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []

        # this buffers store the entire video in memory and are required for fast track exporting
        self.frame_buffer = FrameBuffer()

    def load_cptv(self, filename):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.source_file = filename

        with open(filename, "rb") as f:
            reader = CPTVReader(f)
            local_tz = pytz.timezone("Pacific/Auckland")
            self.video_start_time = reader.timestamp.astimezone(local_tz)
            self.preview_secs = reader.preview_secs
            self.stats.update(self.get_video_stats())
            # we need to load the entire video so we can analyse the background.
            frames = [frame.pix for frame in reader]
            self.frame_buffer.thermal = frames
            edge = self.config.edge_pixels
            self.crop_rectangle = Rectangle(
                edge,
                edge,
                reader.x_resolution - 2 * edge,
                reader.y_resolution - 2 * edge,
            )
def parse_clip(self):
    	    # for now just always calculate as we are using the stats...
        background, background_stats = self.process_background(frames)

        if self.config.background_calc == self.PREVIEW:
            if self.preview_secs > 0:
                self.background_is_preview = True
                background = self.calculate_preview(frames)
            else:
                logging.info(
                    "No preview secs defined for CPTV file - using statistical background measurement"
                )

   	# create optical flow
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setUseInitialFlow(True)
        if not self.config.high_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            self.opt_flow.setTau(1 / 4)
            self.opt_flow.setScalesNumber(3)
            self.opt_flow.setWarpingsNumber(3)
            self.opt_flow.setScaleStep(0.5)

        # process each frame
        self.frame_on = 0
        for frame in frames:
            self._process_frame(frame, background)
            self.frame_on += 1

   def process_background(self, frames):
        background, background_stats = self.analyse_background(frames)
        is_static_background = (
            background_stats.background_deviation
            < self.config.static_background_threshold
        )

        self.stats["threshold"] = background_stats.threshold
        self.stats["average_background_delta"] = background_stats.background_deviation
        self.stats["average_delta"] = background_stats.average_delta
        self.stats["mean_temp"] = background_stats.mean_temp
        self.stats["max_temp"] = background_stats.max_temp
        self.stats["min_temp"] = background_stats.min_temp
        self.stats["is_static"] = is_static_background

        self.threshold = background_stats.threshold

        # if the clip is moving then remove the estimated background and just use a threshold.
        if not is_static_background or self.disable_background_subtraction:
            background = None

        return background, background_stats

    def analyse_background(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background, background_stats
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        frames = np.float32(frames)
        background = np.percentile(frames, q=10, axis=0)
        filtered = np.float32(
            [self.get_filtered(frame, background) for frame in frames]
        )

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(
            frames[:-1], dtype=np.float32
        )
        average_delta = float(np.mean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(
            np.percentile(
                np.reshape(filtered, [-1]), q=self.config.threshold_percentile
            )
        )

        # cap the threshold to something reasonable
        if threshold < self.config.min_threshold:
            threshold = self.config.min_threshold
        if threshold > self.config.max_threshold:
            threshold = self.config.max_threshold

        background_stats = BackgroundAnalysis()
        background_stats.threshold = float(threshold)
        background_stats.average_delta = float(average_delta)
        background_stats.min_temp = float(np.min(frames))
        background_stats.max_temp = float(np.max(frames))
        background_stats.mean_temp = float(np.mean(frames))
        background_stats.background_deviation = float(np.mean(np.abs(filtered)))

        return background, background_stats

    def _process_frame(self, thermal, background=None):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        :param background: (optional) Background image, a numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        thermal = np.float32(thermal)
        filtered = self.get_filtered(thermal, background)

        mask = np.zeros(filtered.shape)
        mask[edge : frame_height - edge, edge : frame_width - edge] = small_mask

        # save frame stats
        self.frame_stats_min.append(np.min(thermal))
        self.frame_stats_max.append(np.max(thermal))
        self.frame_stats_median.append(np.median(thermal))
        self.frame_stats_mean.append(np.mean(thermal))

        # save history
        self.frame_buffer.filtered.append(np.float32(filtered))
        self.frame_buffer.mask.append(np.float32(mask))

    def generate_optical_flow(self):
        if not self.frame_buffer.has_flow:
            self.frame_buffer.generate_optical_flow(
                self.opt_flow, self.config.flow_threshold
            )


     def get_frame_channels(self, region, frame_number):
        """
        Gets frame channels for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown.  Requires the frame_buffer to be filled.
        :param track: the track to get frames for.
        :param frame_number: the frame number where 0 is the first frame of the track.
        :return: numpy array of size [channels, height, width] where channels are thermal, filtered, u, v, mask
        """

		# region_bounds = region_data[1]
		# start_s = region_data[0]
  #       bounds = Region(region_bounds[0],region_bounds[1],width,height)
  #       frame_number = round(start_s * FRAMES_PER_SECOND) 

        if frame_number < 0 or frame_number >= len(self.frame_buffer.thermal):
            raise ValueError(
                "Frame {} is out of bounds for track with {} frames".format(
                    frame_number, len(self.frame_buffer.thermal)
                )
            )

        thermal = bounds.subimage(self.frame_buffer.thermal[frame_number])
        filtered = bounds.subimage(self.frame_buffer.filtered[frame_number])
        flow = bounds.subimage(self.frame_buffer.flow[frame_number])
        mask = bounds.subimage(self.frame_buffer.mask[frame_number])

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we loose a little precision on the filtered frames, but not much (only 1 bit)
        frame = np.int16(
            np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=0)
        )

        return frame

    def start_and_end_time_absolute(self, start,end):
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

 	def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return TrackMovementStatistics()

        # get movement vectors
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        variance_history = [bound.pixel_variance for bound in self.bounds_history]
        mid_x = [bound.mid_x for bound in self.bounds_history]
        mid_y = [bound.mid_y for bound in self.bounds_history]
        delta_x = [mid_x[0] - x for x in mid_x]
        delta_y = [mid_y[0] - y for y in mid_y]
        vel_x = [cur - prev for cur, prev in zip(mid_x[1:], mid_x[:-1])]
        vel_y = [cur - prev for cur, prev in zip(mid_y[1:], mid_y[:-1])]

        movement = sum((vx ** 2 + vy ** 2) ** 0.5 for vx, vy in zip(vel_x, vel_y))
        max_offset = max((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in zip(delta_x, delta_y))

        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        delta_std = float(np.mean(variance_history)) ** 0.5

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        score = min(movement_points, 100) + min(delta_points, 100)

        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
        )

        return stats