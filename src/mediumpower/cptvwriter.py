# Copyright 2019 The Cacophony Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import struct
from io import BytesIO


import struct

MAGIC = b"CPTV"
VERSION = b"\x02"
COLS = 160
ROWS = 120


def write_header(filename, headers,config,timestamp_micros, min_value = 0,max_value=0,num_frames = 0):
    with gzip.open(filename, 'wb', compresslevel=1) as s:
        s.write(MAGIC)
        s.write(VERSION)


        fw = FieldWriter()
        fw.uint16(ord(Field.MIN_VALUE), min_value)
        fw.uint16(ord(Field.MAX_VALUE), max_value)
        fw.uint16(ord(Field.NUM_FRAMES), num_frames)


        fw.uint8(ord(Field.COMPRESSION), 1)
        fw.uint32(ord(Field.X_RESOLUTION), COLS)
        fw.uint32(ord(Field.Y_RESOLUTION), ROWS)

        fw.uint32(ord(Field.DEVICEID), config.device.device_id)
        fw.timestamp(ord(Field.TIMESTAMP), timestamp_micros)
        fw.string(ord(Field.MODEL), headers.model.encode())
        fw.string(ord(Field.BRAND),headers.brand.encode())
        if headers.serial:
            # sometimes sent as None
            fw.uint32(ord(Field.CAMERA_SERIAL),headers.serial)
        fw.string(ord(Field.FIRMWARE),headers.firmware.encode())
        if config.location.latitude != 0:
            fw.float32(ord(Field.LATITUDE), config.location.latitude)
        if config.location.longitude != 0:
            fw.float32(ord(Field.LONGITUDE), config.location.longitude)
        if config.location.altitude:
            fw.float32(ord(Field.ALTITUDE), config.location.altitude)
        if config.location.accuracy:
            fw.float32(ord(Field.ACCURACY), config.location.accuracy)
        fw.write(ord(Section.HEADER), s)
    
class FieldWriter:
    def __init__(self):
        self.s = BytesIO()
        self.count = 0

    def write(self, section_type, dest):
        dest.write(struct.pack("<BB", section_type, self.count))
        dest.write(self.s.getbuffer())

    def timestamp(self, code, t):
        self.uint64(code, t)

    def uint8(self, code, val):
        self.s.write(struct.pack("<BBB", 1, code, val))
        self.count += 1

        
    def uint16(self, code, val):
        self.s.write(struct.pack("<BBH", 2, code, int(val)))
        self.count += 1

    def uint32(self, code, val):
        self.s.write(struct.pack("<BBL", 4, code, int(val)))
        self.count += 1

    def uint64(self, code, val):
        self.s.write(struct.pack("<BBQ", 8, code, val))
        self.count += 1

    def float32(self, code, fval):
        self.s.write(struct.pack("<BBf", 4, code, fval))
        self.count += 1

    def string(self, code, val):
        self.s.write(struct.pack("<BB", len(val), code))
        self.s.write(val)
        self.count += 1


class Section:
    HEADER = b"H"
    FRAME = b"F"


class Field:
    # Header fields
    TIMESTAMP = b"T"
    X_RESOLUTION = b"X"
    Y_RESOLUTION = b"Y"
    COMPRESSION = b"C"
    DEVICENAME = b"D"
    DEVICEID = b"I"

    PREVIEW_SECS = b"P"
    MOTION_CONFIG = b"M"
    LATITUDE = b"L"
    LONGITUDE = b"O"

    LOC_TIMESTAMP = b"S"
    ALTITUDE = b"A"
    ACCURACY = b"U"
    FPS = b"Z"
    MODEL = b"E"
    BRAND = b"B"
    FIRMWARE = b"V"
    CAMERA_SERIAL = b"N"
    BACKGROUND_FRAME = b"g"

    # Frame fields
    BIT_WIDTH = b"w"
    FRAME_SIZE = b"f"
    TIME_ON = b"t"
    LAST_FFC_TIME = b"c"

    TEMP_C = b"a"
    LAST_FFC_TEMP_C = b"b"

    MIN_VALUE = b"Q"
    MAX_VALUE = b"K"
    NUM_FRAMES = b'J'


TIMESTAMP_FIELDS = {Field.TIMESTAMP, Field.LOC_TIMESTAMP}

UINT32_FIELDS = {
    Field.X_RESOLUTION,
    Field.Y_RESOLUTION,
    Field.FRAME_SIZE,
    Field.TIME_ON,
    Field.LAST_FFC_TIME,
    Field.DEVICEID,
    Field.CAMERA_SERIAL,
}

UINT8_FIELDS = {
    Field.COMPRESSION,
    Field.BIT_WIDTH,
    Field.PREVIEW_SECS,
    Field.FPS,
    Field.BACKGROUND_FRAME,
}

STRING_FIELDS = {
    Field.DEVICENAME,
    Field.MOTION_CONFIG,
    Field.MODEL,
    Field.BRAND,
    Field.FIRMWARE,
}

FLOAT_FIELDS = {
    Field.LATITUDE,
    Field.LONGITUDE,
    Field.ALTITUDE,
    Field.ACCURACY,
    Field.LAST_FFC_TEMP_C,
    Field.TEMP_C,
}
