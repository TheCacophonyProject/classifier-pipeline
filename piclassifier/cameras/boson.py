from .rawframe import RawFrame


class Boson(RawFrame):
    VOSPI_DATA_SIZE = 160
    TELEMETRY_PACKET_COUNT = 4

    def __init__(self, headers):
        super().__init__(headers)
