from cptv import CPTVWriter
from datetime import datetime
import os
import yaml

CPTV_TEMP_EXT = "cptv.temp"


class CPTVRecorder:
    def __init__(self, location_config, motion_config):
        self.location_config = location_config
        self.motion_config = motion_config
        self.writer = None
        self.filename = None
        self.recording = False

    def start_recording(self, clip):
        self.filename = new_temp_name()
        self.filename = os.path.join(self.motion_config.output_dir, self.filename)

        f = open(self.filename, "wb")
        self.writer = CPTVWriter(self.filename)
        self.writer.timestamp = datetime.now()
        self.writer.device_name = b"gp-test-01"
        self.writer.latitude = self.location_config.latitude
        self.writer.longitude = self.location_config.longitude
        self.writer.preview_secs = self.motion_config.preview_secs
        self.writer.motion_config = b""
        self.writer.write_header()

        self.recording = True

    def write_frame(self, frame):
        self.writer.write_frame(frame)

    def stop_recording(self):
        self.writer.close()
        final_name = os.path.splitext(self.filename)[0]
        os.rename(self.filename, final_name)
        self.writer = None
        self.recording = False

    def delete_recording(self):
        self.writer.close()
        os.remove(self.filename)
        self.writer = None
        self.recording = False

def new_temp_name():
    return datetime.now().strftime("%Y%m%d.%H%M%S.%f" + CPTV_TEMP_EXT)
