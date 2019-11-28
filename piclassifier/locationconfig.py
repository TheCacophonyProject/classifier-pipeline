from pathlib import Path
import attr
import yaml

CONFIG_FILENAME = "location.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


@attr.s
class LocationConfig:

    DEFAULT_LAT = -43.5321
    DEFAULT_LONG = 172.6362

    latitude = attr.ib()
    longitude = attr.ib()
    loc_timestamp = attr.ib()
    altitude = attr.ib()
    accuracy = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = LocationConfig.find_config()
        with open(filename) as stream:
            return cls.load_from_stream(stream)

    @classmethod
    def load_from_stream(cls, stream):
        raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
        return cls(
            latitude=raw.get("latitude", 0),
            longitude=raw.get("longitude", 0),
            loc_timestamp=raw.get("timestamp"),
            altitude=raw.get("altitude"),
            accuracy=raw.get("accuracy"),
        )

    def get_lat_long(self, use_default=False):
        lat = self.latitude
        lng = self.longitude
        if use_default and lat == 0:
            lat = LocationConfig.DEFAULT_LAT
        if use_default and lng == 0:
            lng == LocationConfig.DEFAULT_LONG
        return (lat, lng)

    @staticmethod
    def find_config():
        for directory in CONFIG_DIRS:
            p = directory / CONFIG_FILENAME
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            "No configuration file found.  Looking for file named '{}' in dirs {}".format(
                CONFIG_FILENAME, CONFIG_DIRS
            )
        )
