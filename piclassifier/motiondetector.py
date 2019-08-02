from datetime import datetime, timedelta
from astral import Astral, Location
import numpy as np
import timezonefinder, pytz
from ml_tools import tools
from ml_tools.tools import Rectangle


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None

    @property
    def current(self):
        if self.last_index is not None:
            return self.frames[self.last_index]
        return None

    def get(self, i):
        i = i % self.size
        return self.frames[i]

    @property
    def oldest(self):
        if self.oldest_index is not None:
            return self.frames[self.oldest_index]
        return None

    def add(self, frame):
        if self.last_index is None:
            self.last_index = 0
            self.oldest_index = 0
            self.frames[0] = frame
        else:
            self.last_index = (self.last_index + 1) % self.size
            if self.last_index == self.oldest_index:
                self.oldest_index = (self.oldest_index + 1) % self.size
        self.frames[self.last_index] = frame


class MotionDetector:
    def __init__(self, res_x, res_y, config, location_config, dynamic_thresh):
        # self.preview_frames_count = motion_config.preview_frames
        self.config = config
        self.location_config = location_config
        self.preview_frames = config.preview_secs * config.frame_rate
        self.compare_gap = config.frame_compare_gap + 1
        edge = config.edge_pixels

        self.clipped_window = SlidingWindow(
            (self.compare_gap, res_y - edge * 2, res_x - edge * 2), np.int32
        )
        self.diff_window = SlidingWindow(
            (self.compare_gap, res_y - edge * 2, res_x - edge * 2), np.int32
        )

        self.thermal_window = SlidingWindow(
            (self.preview_frames, res_y, res_x), np.uint16
        )
        self.processed = 0
        self.num_frames = 0
        self.thermal_thresh = 0
        self.background = None
        self.movement_detected = False
        self.dynamic_thresh = dynamic_thresh
        self.temp_thresh = config.temp_thresh

        self.crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)

        self.last_sunrise_check = None
        self.location = None
        self.use_sunrise = self.config.use_sunrise_sunset
        self.sunrise = None
        self.sunset = None
        self.recording = False
        if self.use_sunrise:
            self.set_location(location_config)

    def set_location(self, location_config):
        self.location = Location()
        self.location.latitude = location_config.latitude
        self.location.longitude = location_config.longitude
        print(self.location)

        self.location.altitude = location_config.altitude
        self.location.timezone = tools.get_timezone_str(
            location_config.latitude, location_config.longitude
        )

    def set_sunrise_sunet(self):
        date = datetime.now().date()
        if self.last_sunrise_check is None or date > self.last_sunrise_check:
            sun = self.location.sun()

            self.sunrise = (
                sun["sunrise"] + timedelta(minutes=self.config.sunrise_offset)
            ).time()
            self.sunset = (
                sun["sunset"] + timedelta(minutes=self.config.sunset_offset)
            ).time()
            self.last_sunrise_check = date
            print(
                "sunrise is {} sunset is {} next check is {}".format(
                    self.sunrise, self.sunset, self.last_sunrise_check
                )
            )

    def calc_temp_thresh(self, thermal_frame):
        if self.dynamic_thresh:
            self.background = np.minimum(self.background, thermal_frame)
            self.temp_thresh = min(self.config.temp_thresh, np.average(self.background))
        else:
            self.temp_thresh = self.config.temp_thresh

    def detect(self, clipped_frame):
        oldest = self.clipped_window.oldest

        delta_frame = clipped_frame - oldest

        if not self.config.warmer_only:
            delta_frame = abs(delta_frame)
        if self.config.one_diff_only:
            diff = len(delta_frame[delta_frame >= self.config.delta_thresh])
        else:
            if self.processed > 2:
                delta_frame2 = self.diff_window.oldest

                # delta_frame2[
                #     delta_frame2 >= self.config.delta_thresh
                # ] = self.config.delta_thresh

                delta_frame[
                    delta_frame >= self.config.delta_thresh
                ] = self.config.delta_thresh

                delta_frame = delta_frame2 + delta_frame
                diff = len(delta_frame[delta_frame == self.config.delta_thresh * 2])
            else:
                delta_frame[
                    delta_frame >= self.config.delta_thresh
                ] = self.config.delta_thresh

        self.diff_window.add(delta_frame)
        if diff > self.config.count_thresh:
            print("movement detected at frame {}".format(self.num_frames))
            return True
        return False

    def can_record(self):
        if self.use_sunrise:
            self.set_sunrise_sunet()
            time = datetime.now().time()
            return time > self.sunset or time < self.sunrise
        return True

    def start_recording(self):
        self.recoridng = True

    def stop_recording(self):
        self.recording = False

    def process_frame(self, thermal_frame):
        if self.can_record() or self.recording:
            if self.num_frames < 9 * 10:
                self.num_frames += 1
                return
            frame = np.int32(self.crop_rectangle.subimage(thermal_frame))
            clipped_frame = np.clip(np.int32(frame), self.config.temp_thresh, None)
            self.clipped_window.add(clipped_frame)
            self.thermal_window.add(thermal_frame)
            if self.processed == 0:
                self.background = thermal_frame
            else:
                self.calc_temp_thresh(thermal_frame)
                self.movement_detected = self.detect(clipped_frame)
            self.processed += 1
        else:
            self.movement_detected = False
        self.num_frames += 1
