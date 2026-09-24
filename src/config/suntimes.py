"""Sunrise and sunset times using the NOAA solar calculations (as used by astral),
without the import cost of astral"""

from datetime import date as date_type, datetime, timedelta, timezone
from math import acos, asin, cos, degrees, radians, sin, sqrt, tan

EARTH_RADIUS = 6356900
# sun's centre is 0.833 degrees below the horizon at sunrise / sunset
SUNRISE_DEPRESSION = 90.833


def _julian_century(date):
    julian_day = date.toordinal() - date_type(1900, 1, 1).toordinal() + 2 + 2415018.5
    return (julian_day - 2451545.0) / 36525.0


def _depression_adjustment(elevation):
    # extra degrees of depression due to observer elevation in metres
    if not elevation or elevation <= 0:
        return 0
    theta = acos(EARTH_RADIUS / (EARTH_RADIUS + elevation))
    a = EARTH_RADIUS * sin(theta)
    b = EARTH_RADIUS - EARTH_RADIUS * cos(theta)
    return degrees(acos(a / sqrt(a * a + b * b)))


def _sun_position(jc):
    # returns (equation of time in minutes, sun declination in degrees)
    mean_long = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360.0
    mean_anomaly = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccentricity = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    m = radians(mean_anomaly)
    eq_of_center = (
        sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + sin(2 * m) * (0.019993 - 0.000101 * jc)
        + sin(3 * m) * 0.000289
    )
    omega = radians(125.04 - 1934.136 * jc)
    apparent_long = mean_long + eq_of_center - 0.00569 - 0.00478 * sin(omega)

    seconds = 21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))
    mean_obliquity = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliquity = mean_obliquity + 0.00256 * cos(omega)

    declination = degrees(asin(sin(radians(obliquity)) * sin(radians(apparent_long))))

    y = tan(radians(obliquity) / 2.0) ** 2
    l0 = radians(mean_long)
    e = eccentricity
    eq_of_time = (
        y * sin(2 * l0)
        - 2.0 * e * sin(m)
        + 4.0 * e * y * sin(m) * cos(2 * l0)
        - 0.5 * y * y * sin(4 * l0)
        - 1.25 * e * e * sin(2 * m)
    )
    return degrees(eq_of_time) * 4.0, declination


def _calc_time(date, lat, lng, elevation, rising):
    lat = max(min(lat, 89.8), -89.8)
    eq_of_time, declination = _sun_position(_julian_century(date))

    depression = radians(SUNRISE_DEPRESSION + _depression_adjustment(elevation))
    lat_rad = radians(lat)
    dec_rad = radians(declination)
    h = cos(depression) / (cos(lat_rad) * cos(dec_rad)) - tan(lat_rad) * tan(dec_rad)
    if h < -1 or h > 1:
        raise ValueError("Sun never reaches the horizon on this day, at this location")
    hour_angle = acos(h)
    if not rising:
        hour_angle = -hour_angle

    minutes_utc = 720.0 - 4.0 * (lng + degrees(hour_angle)) - eq_of_time
    day = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    return day + timedelta(seconds=int(minutes_utc * 60))


def sun_times(date, lat, lng, elevation=0):
    """Returns timezone aware UTC sunrise and sunset datetimes for date"""
    return {
        "sunrise": _calc_time(date, lat, lng, elevation, True),
        "sunset": _calc_time(date, lat, lng, elevation, False),
    }
