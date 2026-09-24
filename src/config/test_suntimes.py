import pytest
from datetime import date, datetime, timezone

from .suntimes import sun_times


# expected values generated with astral 1.10.1 Astral().sun_utc
@pytest.mark.parametrize(
    "day,lat,lng,elevation,sunrise,sunset",
    [
        (
            date(2026, 9, 24),
            -43.5321,
            172.6362,
            0,
            datetime(2026, 9, 23, 18, 15, 37, tzinfo=timezone.utc),
            datetime(2026, 9, 24, 6, 27, 43, tzinfo=timezone.utc),
        ),
        (
            date(2026, 6, 21),
            -36.85,
            174.76,
            100,
            datetime(2026, 6, 20, 19, 32, 49, tzinfo=timezone.utc),
            datetime(2026, 6, 21, 5, 12, 32, tzinfo=timezone.utc),
        ),
        (
            date(2026, 12, 21),
            51.5,
            -0.12,
            0,
            datetime(2026, 12, 21, 8, 3, 29, tzinfo=timezone.utc),
            datetime(2026, 12, 21, 15, 53, 7, tzinfo=timezone.utc),
        ),
    ],
)
def test_sun_times(day, lat, lng, elevation, sunrise, sunset):
    times = sun_times(day, lat, lng, elevation)
    assert times["sunrise"] == sunrise
    assert times["sunset"] == sunset


def test_no_sunset():
    # midsummer in the arctic
    with pytest.raises(ValueError):
        sun_times(date(2026, 6, 21), 80, 0)
