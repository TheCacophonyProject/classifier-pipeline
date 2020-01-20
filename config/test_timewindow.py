import pytest
from datetime import datetime, timedelta

from .timewindow import RelAbsTime, TimeWindow


class TestRelAbs:
    def test_defaults(self):
        rel_time = RelAbsTime(None)
        assert rel_time.is_after()
        assert rel_time.is_before()

        cur_date = datetime.now()

        rel_time = RelAbsTime(cur_date.strftime("%H:%M"))
        assert rel_time.is_after()
        assert not rel_time.is_before()

        rel_time = RelAbsTime(None, default_time=cur_date.time())
        assert rel_time.is_after()
        assert not rel_time.is_before()
        assert rel_time.offset_s is None

    def test_offset(self):
        cur_date = datetime.now()
        rel_time = RelAbsTime("30m", default_time=cur_date.time())
        assert rel_time.offset_s == 30 * 60
        rel_time = RelAbsTime("30s", default_offset=200)
        assert rel_time.offset_s == 30
        rel_time = RelAbsTime("1.5h")
        assert rel_time.offset_s == 1.5 * 60 * 60
        rel_time = RelAbsTime("1a.d5h", default_time=cur_date.time())
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

    def test_absolute_times(self):
        cur_date = datetime.now()
        start = RelAbsTime(cur_date.strftime("%H:%M"))
        end = RelAbsTime(cur_date.strftime("%H:%M"))
        time_window = TimeWindow(start, end)
        assert not time_window.inside_window()

        time_window.end = RelAbsTime((cur_date + timedelta(hours=1)).strftime("%H:%M"))
        assert time_window.inside_window()

        new_end = cur_date + timedelta(minutes=-1)
        if(new_end.day < cur_date.day):
            new_end = cur_date
        time_window.end = RelAbsTime(new_end.strftime("%H:%M"))
        assert not time_window.inside_window()

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

        time_window.end.time = (cur_date + timedelta(hours=11)).time()
        assert time_window.inside_window()
