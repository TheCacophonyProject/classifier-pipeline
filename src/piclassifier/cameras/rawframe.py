from abc import ABC, abstractmethod
from struct import unpack_from
import numpy as np
from cptv import Frame


class RawFrame(ABC):
    def __init__(self, headers):
        self.pix = None
        self.telemetry = None
        self.res_x = headers.res_x
        self.res_y = headers.res_y
        self.img_dtype = np.dtype("uint{}".format(headers.pixel_bits))
        self.received_at = None

    def parse(self, data):
        telemetry = self.parse_telemetry(data[: self.get_telemetry_size()])

        thermal_frame = np.frombuffer(
            data, dtype=self.img_dtype, offset=self.get_telemetry_size()
        ).reshape(self.res_y, self.res_x)
        return Frame(
            thermal_frame.byteswap(),
            telemetry.time_on,
            telemetry.last_ffc_time,
            telemetry.fpa_temp,
            telemetry.fpa_temp_last_ffc,
        )

    @abstractmethod
    def get_telemetry_size(self): ...

    @abstractmethod
    def parse_telemetry(self, raw_bytes): ...


def get_uint16(raw, offset):
    return unpack_from(">H", raw, offset)[0]


def get_uint32(raw, offset):
    return (
        raw[offset + 1]
        | (raw[offset] << 8)
        | (raw[offset + 3] << 16)
        | (raw[offset + 2] << 24)
    )


def get_uint64(raw, offset):
    return (
        raw[offset + 1]
        | (raw[offset] << 8)
        | (raw[offset + 3] << 16)
        | (raw[offset + 2] << 24)
        | (raw[offset + 5] << 32)
        | (raw[offset + 4] << 40)
        | (raw[offset + 7] << 48)
        | (raw[offset + 6] << 56)
    )
