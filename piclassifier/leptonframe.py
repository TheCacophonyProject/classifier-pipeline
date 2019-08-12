from struct import unpack_from


class LeptonFrame:
    def __init__(self, telemetry, frame):
        self.telemetry = telemetry
        self.pix = frame


class Telemetry:
    def __init__(self):
        self.telemetry_revision = None
        self.time_on = None
        self.status_bits = None
        self.reserved5 = None
        self.software_revision = None
        self.reserved17 = None
        self.frame_counter = None
        self.frame_mean = None
        self.fpa_temp_counts = None
        self.fpa_temp = None
        self.reserved25 = None
        self.fpa_temp_last_ffc = None
        self.last_ffc_time = None

    @classmethod
    def parse_telemetry(cls, raw_bytes):
        offset = 0
        revision = get_uint16(raw_bytes, offset)
        time_counter = get_uint32(raw_bytes, 2)
        status_bits = get_uint32(raw_bytes, 6)

        offset = 2 * (1 + 2 + 2 + 8)
        software_revision = get_uint64(raw_bytes, offset)  # /26
        offset += 2 * (4 + 3)
        frame_counter = get_uint32(raw_bytes, offset)
        offset += 4
        frame_mean, fpa_temp_counts, fpa_temp = unpack_from(
            ">HHH", raw_bytes, offset=offset
        )

        offset += 2 * (1 + 1 + 1 + 4)
        fpa_temp_last_ffc = get_uint16(raw_bytes, offset)
        offset += 2
        time_counter_last_ffc = get_uint32(raw_bytes, offset)
        # 60th byte
        t = cls()
        t.telemetry_revision = revision
        t.time_on = time_counter
        t.status_bits = status_bits
        t.software_revision = software_revision
        t.frame_counter = frame_counter
        t.frame_mean = frame_mean
        t.fpa_temp_counts = fpa_temp_counts
        t.fpa_temp = fpa_temp
        t.fpa_temp_last_ffc = fpa_temp_last_ffc
        t.last_ffc_time = time_counter_last_ffc
        return t


def get_uint16(raw, offset):
    return unpack_from(">I", raw, offset)


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
