from datetime import timedelta
from .rawframe import RawFrame
from piclassifier.telemetry import Telemetry


class IRFrame(RawFrame):
    def __init__(self, headers):
        super().__init__(headers)

    def get_telemetry_size(self):
        return 0

    def parse_telemetry(self, raw_bytes):
        t = Telemetry()
        return t
