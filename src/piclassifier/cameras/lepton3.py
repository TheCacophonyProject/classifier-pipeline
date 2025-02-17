from struct import unpack_from
from datetime import timedelta
from .rawframe import RawFrame, get_uint32, get_uint16, get_uint64
from piclassifier.telemetry import Telemetry

import struct


class Lepton3(RawFrame):
    VOSPI_DATA_SIZE = 160
    TELEMETRY_PACKET_COUNT = 4

    def __init__(self, headers):
        super().__init__(headers)

    def get_telemetry_size(self):
        return Lepton3.VOSPI_DATA_SIZE * Lepton3.TELEMETRY_PACKET_COUNT

    def parse_telemetry(self, raw_bytes):
        offset = 0
        revision = get_uint16(raw_bytes, offset)
        time_counter = get_uint32(raw_bytes, 2)
        status_bits = get_uint32(raw_bytes, 6)
        ffc_status = (status_bits >> 4) & 0b11
        offset = 2 + 4 + 4 + 16
        software_revision = get_uint64(raw_bytes, offset)  # /26
        offset += 8 + 6
        frame_counter = get_uint32(raw_bytes, offset)
        offset += 4

        frame_mean, fpa_temp_counts, fpa_temp = unpack_from(
            ">HHH", raw_bytes, offset=offset
        )
        offset += 2 * (1 + 1 + 1 + 4)
        fpa_temp_last_ffc = get_uint16(raw_bytes, offset)

        offset += 2
        time_counter_last_ffc = get_uint32(raw_bytes, offset)
        ffc_imminent = raw_bytes[637] > 0

        # 60th byte
        t = Telemetry()
        t.telemetry_revision = revision
        t.time_on = timedelta(milliseconds=time_counter)
        t.status_bits = status_bits
        t.software_revision = software_revision
        t.frame_counter = frame_counter
        t.frame_mean = frame_mean
        t.fpa_temp_counts = fpa_temp_counts
        t.fpa_temp = (fpa_temp - 27315.0) / 100
        t.fpa_temp_last_ffc = (fpa_temp_last_ffc - 27315.0) / 100
        t.last_ffc_time = timedelta(milliseconds=time_counter_last_ffc)
        t.ffc_imminent = ffc_imminent
        t.ffc_status = ffc_status
        return t

    # not used for anything at the moment,  has not been tested
    def raw_telemetry(self):
        telemetry_data = b""
        telemetry_data += pack_uint16(self.telemetry.telemetry_revision)
        telemetry_data += pack_uint32(self.telemetry.timeon.milliseconds)
        status_bits = self.telemetry.status_bits << 4
        telemetry_data += pack_uint32(status_bits)

        telemetry_data += pack_uint64(self.telemetry.software_revision)

        telemetry_data += pack_uint32(self.telemetry.frame_counter)

        telemetry_data += pack_uint16(self.telemetry.frame_mean)
        telemetry_data += pack_uint16(self.telemetry.fpa_temp_counts)
        telemetry_data += pack_uint16(self.telemetry.fpa_temp * 100 + 27315)

        telemetry_data += pack_uint16(self.telemetry.fpa_temp_last_ffc * 100 + 27315)
        telemetry_data += pack_uint32(self.telemetry.last_ffc_time.milliseconds)
        remaining = self.get_telemetry_size() - len(telemetry_data)
        telemetry_data += bytearray(remaining)

        return telemetry_data

    def raw_data(self):
        telemetry_bytes = self.raw_telemetry
        frame_bytes = self.pix.tobytes()
        return telemetry_bytes + frame_bytes


def pack_uint16(data):
    return struct.pack(">H", data)


def pack_uint64(data):
    return struct.pack(">L", data)


def pack_uint32(data):
    return struct.pack(">L", data)
