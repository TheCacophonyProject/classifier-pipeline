""" A window time frame, which can be relative to sunset and sunrise
"""

from datetime import datetime, timedelta
import logging
import enum
from astral import Location

from ml_tools import tools


class WindowStatus(enum.Enum):
    """Types of frames"""

    before = 0
    inside = 1
    after = 2
    non_stop = 3


class TimeWindow:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.location = None
        self.last_sunrise_check = None
        self.non_stop = not self.use_sunrise_sunset() and self.start.dt == self.end.dt

    def clone(self):
        new_window = TimeWindow(self.start.clone(), self.end.clone())
        new_window.location = self.location
        new_window.last_sunrise_check = self.last_sunrise_check
        new_window.non_stop = self.non_stop
        return new_window

    def next_start(self):
        return next_time(self.start)

    def next_end(self):
        return next_time(self.end)

    def use_sunrise_sunset(self):
        return self.start.is_relative or self.end.is_relative

    def window_status(self):
        if self.non_stop:
            return WindowStatus.non_stop
        if self.use_sunrise_sunset():
            self.update_sun_times()
        if self.start.is_before():
            return WindowStatus.before
        if self.end.is_before():
            return WindowStatus.inside
        return WindowStatus.after

    def next_window(self):
        if self.use_sunrise_sunset():
            # update to tomorrows times
            self.update_sun_times(True)
        else:
            if self.start.dt is not None:
                self.start.dt = self.start.dt + timedelta(days=1)
            if self.end.dt is not None:
                self.end.dt = self.end.dt + timedelta(days=1)
        logging.info(
            "Updated to next window start %s end %s", self.start.dt, self.end.dt
        )

    def inside_window(self):
        if self.use_sunrise_sunset():
            self.update_sun_times()
            if self.end.dt < self.start.dt:
                return self.start.is_after() or self.end.is_before()
            elif self.start.is_after() and self.end.is_after():
                self.next_window()
                return False
            elif self.end.is_before() and self.end.dt.date() == datetime.now().date():
                # overnight window and we are before sunrise
                return True
        elif self.start.time == self.end.time:
            return True
        if self.start.is_after() and self.end.is_after():
            self.next_window()
            return False
        return self.start.is_after() and self.end.is_before()

    def update_sun_times(self, next_window=False):
        if not self.use_sunrise_sunset():
            return

        if self.location is None:
            raise ValueError(
                "Location must be set for relative times, by calling set_location"
            )
        # if next_window:
        # date = date + timedelta(days=1)
        # after_window = self.start.is_before() or self.end.is_before()
        # if self.last_sunrise_check is not None and (
        #     self.start.is_before() or self.end.is_before()
        # ):
        if (
            self.last_sunrise_check is None
            or next_window
            or datetime.now() > self.end.dt
        ):
            #     return
            date = datetime.now().date()
            if self.last_sunrise_check is not None and next_window:
                date = self.last_sunrise_check + timedelta(days=1)
            elif next_window:
                date = date + timedelta(days=1)
            sun_times = self.location.sun(date=date)
            self.last_sunrise_check = date
            if self.start.is_relative:
                self.start.dt = sun_times["sunset"] + timedelta(
                    seconds=self.start.offset_s
                )
                self.start.dt = self.start.dt.replace(tzinfo=None)

            if self.end.is_relative:
                self.end.dt = sun_times["sunrise"] + timedelta(
                    seconds=self.end.offset_s
                )
                self.end.dt = self.end.dt.replace(tzinfo=None)
                if datetime.now() > self.end.dt and self.end.dt < self.start.dt:
                    date = date + timedelta(days=1)
                    sun_times = self.location.sun(date=date)
                    self.end.dt = sun_times["sunrise"] + timedelta(
                        seconds=self.end.offset_s
                    )
                    self.end.dt = self.end.dt.replace(tzinfo=None)

            logging.info(
                "Updated sun times start is {} end is {}".format(
                    self.start.dt, self.end.dt
                )
            )

    def set_location(self, lat, lng, altitude=0):
        self.location = Location()
        self.location.latitude = lat
        self.location.longitude = lng
        self.location.altitude = altitude
        self.location.timezone = tools.get_timezone_str(lat, lng)

        self.update_sun_times()


class RelAbsTime:
    def __init__(self, time_str, default_offset=None, default_time=None):
        self.is_relative = False
        self.offset_s = None
        self.dt = None
        self.any_time = False
        if time_str == "" or (
            time_str is None and default_offset is None and default_time is None
        ):
            self.any_time = True
            return

        try:
            self.dt = datetime.combine(
                datetime.now(), datetime.strptime(time_str, "%H:%M").time()
            )
        except (ValueError, TypeError):
            if not time_str:
                self.offset_s = default_offset
            elif isinstance(time_str, int) or time_str.isnumeric():
                self.offset_s = int(time_str)
            else:
                self.offset_s = self.parse_duration(time_str, default_offset)

            if self.offset_s is None and default_time:
                self.dt = default_time
            else:
                self.is_relative = True

    def clone(self):
        new_time = RelAbsTime("")
        new_time.is_relative = self.is_relative
        new_time.offset_s = self.offset_s
        new_time.dt = self.dt
        new_time.any_time = self.any_time
        return new_time

    @property
    def time(self):
        return self.dt.time() if self.dt is not None else None

    def is_after(self, ignore_date=False):
        if ignore_date:
            return self.any_time or datetime.now().time() > self.dt.time()
        return self.any_time or datetime.now() > self.dt

    def is_before(self, ignore_date=False):
        if ignore_date:
            return self.any_time or datetime.now().time() < self.dt.time()
        return self.any_time or datetime.now() < self.dt

    def parse_duration(self, time_str, default_offset=None):
        if not time_str:
            return default_offset

        time_str = time_str.strip()
        unit = time_str[-1]
        if unit.isalpha():
            try:
                offset = float(time_str[:-1])
            except ValueError:
                return default_offset
            if unit == "s":
                return offset
            elif unit == "m":
                return offset * 60
            elif unit == "h":
                return offset * 60 * 60
            return offset

        try:
            offset = float(time_str)
            return offset * 60
        except ValueError:
            pass
        return default_offset


def next_time(rel_time):
    if rel_time.any_time:
        return None
    if rel_time.is_relative:
        return rel_time.dt
    date = rel_time.dt
    return date
