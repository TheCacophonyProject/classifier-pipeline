import attr


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
    def load(cls, raw):
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
            lng = LocationConfig.DEFAULT_LONG
        return (lat, lng)
