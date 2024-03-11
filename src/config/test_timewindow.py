import pytest
from datetime import datetime, timedelta

from .timewindow import RelAbsTime, TimeWindow
from freezegun import freeze_time


class TestRelAbs:
    def test_defaults(self):
        rel_time = RelAbsTime(None)
        assert rel_time.is_after()
        assert rel_time.is_before()

        cur_date = datetime.now()

        rel_time = RelAbsTime(cur_date.strftime("%H:%M"))
        assert rel_time.is_after()
        assert not rel_time.is_before()

        rel_time = RelAbsTime(None, default_time=cur_date)
        assert rel_time.is_after()
        assert not rel_time.is_before()
        assert rel_time.offset_s is None

    def test_offset(self):
        cur_date = datetime.now()
        rel_time = RelAbsTime("30m", default_time=cur_date)
        assert rel_time.offset_s == 30 * 60
        rel_time = RelAbsTime("30s", default_offset=200)
        assert rel_time.offset_s == 30
        rel_time = RelAbsTime("1.5h")
        assert rel_time.offset_s == 1.5 * 60 * 60
        rel_time = RelAbsTime("1a.d5h", default_time=cur_date)
        assert rel_time.offset_s is None
        assert rel_time.time == cur_date.time()

        rel_time = RelAbsTime("3.5")
        assert rel_time.offset_s == 3.5 * 60
        rel_time = RelAbsTime("3.5z")
        assert rel_time.offset_s == 3.5


class TestWindow:
    # christchurch
    DEFAULT_LAT = -43.5321
    DEFAULT_LONG = 172.6362

    @freeze_time(lambda: datetime.now().replace(hour=1, minute=59))
    def test_before_sunrise(self):
        start = RelAbsTime("-30m")
        end = RelAbsTime("30m")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        assert time_window.inside_window()

    @freeze_time(lambda: datetime.now().replace(hour=23, minute=59))
    def test_after_sunset(self):
        start = RelAbsTime("-30m")
        end = RelAbsTime("30m")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        assert time_window.inside_window()

    @freeze_time(
        lambda: (datetime.now() - timedelta(days=1)).replace(hour=23, minute=59)
    )
    def test_after_midnight(self):
        start = RelAbsTime("-30m")
        end = RelAbsTime("30m")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        assert time_window.inside_window()
        with freeze_time(
            (datetime.now() + timedelta(days=1)).replace(hour=1, minute=59)
        ):
            assert time_window.inside_window()

    @freeze_time(lambda: datetime.now().replace(hour=1, minute=59))
    def test_dt_update(self):
        start = RelAbsTime("-30m")
        end = RelAbsTime("30m")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        assert time_window.inside_window()
        end_date = time_window.end.dt
        start_date = time_window.start.dt

        # check that if we move out of window, the end time gets updated
        with freeze_time(datetime.now().replace(hour=12, minute=59)):
            time_window.inside_window()
            assert end_date != time_window.end.dt
            assert start_date == time_window.start.dt

    @freeze_time(lambda: datetime.now().replace(hour=12, minute=1))
    def test_after_sunrise(self):
        start = RelAbsTime("-30m")
        end = RelAbsTime("30m")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        assert not time_window.inside_window()

    def test_absolute_times(self):
        cur_date = datetime.now()
        start = RelAbsTime(cur_date.strftime("%H:%M"))
        end = RelAbsTime(cur_date.strftime("%H:%M"))
        time_window = TimeWindow(start, end)
        assert time_window.inside_window()

        new_end = cur_date + timedelta(minutes=1)
        if new_end.day < cur_date.day:
            new_end = cur_date
        time_window.end = RelAbsTime(new_end.strftime("%H:%M"))
        assert time_window.inside_window()

        new_end = cur_date + timedelta(minutes=-1)
        if new_end.day < cur_date.day:
            new_end = cur_date
        time_window.end = RelAbsTime(new_end.strftime("%H:%M"))
        assert not time_window.inside_window()

    @freeze_time(lambda: datetime.now().replace(hour=12, minute=30))
    def test_sunrise_times(self):
        cur_date = datetime.now()
        start = RelAbsTime(cur_date.strftime("%H:%M"))
        end = RelAbsTime("0s")
        time_window = TimeWindow(start, end)
        with pytest.raises(ValueError):
            time_window.inside_window()
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        time_window.inside_window()
        assert time_window.last_sunrise_check is not None

        time_window.end.dt = cur_date + timedelta(hours=11)
        assert time_window.inside_window()

        start = RelAbsTime(cur_date.strftime("0s"))
        end = RelAbsTime("0s")
        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        time_window.update_sun_times()
        assert time_window.start.dt.date() == datetime.now().date()
        assert time_window.end.dt > time_window.start.dt
        assert time_window.end.dt.date() == datetime.now().date() + timedelta(days=1)
        prev_s = time_window.start.dt.date()
        time_window.update_sun_times()
        assert time_window.start.dt.date() == prev_s

        time_window.update_sun_times(True)
        assert time_window.start.dt.date() == prev_s + timedelta(days=1)

        time_window.inside_window()
        assert time_window.start.dt.date() == prev_s + timedelta(days=1)

        time_window = TimeWindow(start, end)
        time_window.set_location(TestWindow.DEFAULT_LAT, TestWindow.DEFAULT_LONG, 0)
        time_window.update_sun_times()
        prev_s = time_window.start.dt.date()
        time_window.inside_window()

        time_window.start.dt = datetime.now() - timedelta(days=1)
        time_window.end.dt = datetime.now() - timedelta(days=1)
        prev_s = time_window.start.dt.date()

        # current time is after start and end so sould pick tomorrows sunset window
        in_w = time_window.inside_window()
        assert prev_s != time_window.start.dt.date()
