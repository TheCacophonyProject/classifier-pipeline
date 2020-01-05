from cptv import CPTVWriter
from datetime import datetime
import os
import yaml

CPTV_TEMP_EXT = ".cptv.temp"


class CPTVRecorder:
    def __init__(self, thermal_config):
        self.location_config = thermal_config.location
        self.device_config = thermal_config.device
        self.output_dir = thermal_config.recorder.output_dir
        self.motion_config = thermal_config.motion
        self.preview_secs = thermal_config.recorder.preview_secs
        self.writer = None
        self.filename = None
        self.recording = False
        self.frames = 0
        self.device_config = thermal_config.device
        self.min_frames = (
            thermal_config.recorder.min_secs * thermal_config.recorder.frame_rate
        )
        self.max_frames = (
            thermal_config.recorder.max_secs * thermal_config.recorder.frame_rate
        )
        self.write_until = 0

    def force_stop(self):
        if not self.recording:
            return

        if self.has_minimum():
            self.stop_recording()
        else:
            self.delete_recording()

    def process_frame(self, movement_detected, lepton_frame, temp_thresh):
        if movement_detected:
            self.write_until = self.frames + self.min_frames
            self.write_frame(lepton_frame, temp_thresh)
        elif self.recording:
            if self.has_minimum():
                self.stop_recording()
            else:
                self.write_frame(lepton_frame, temp_thresh)

        if self.frames == self.max_frames:
            self.stop_recording()

    def has_minimum(self):
        return self.frames > self.write_until

    def start_recording(self, temp_thresh):
        self.frames = 0
        self.filename = new_temp_name()
        self.filename = os.path.join(self.output_dir, self.filename)
        f = open(self.filename, "wb")
        self.writer = CPTVWriter(f)
        self.writer.timestamp = datetime.now()
        self.writer.latitude = self.location_config.latitude
        self.writer.longitude = self.location_config.longitude
        self.writer.preview_secs = self.preview_secs
        default_thresh = self.motion_config.temp_thresh
        self.motion_config.temp_thresh = temp_thresh
        self.writer.motion_config = yaml.dump(self.motion_config).encode()[:255]
        self.motion_config.temp_thresh = default_thresh

        if self.device_config.name:
            self.writer.device_name = self.device_config.name.encode()
        if self.device_config.device_id:
            self.writer.device_id = self.device_config.device_id

        self.writer.write_header()
        self.recording = True

    def write_frame(self, lepton_frame, temp_thresh):
        if self.writer is None:
            self.start_recording(temp_thresh)
        self.writer.write_frame(lepton_frame)
        self.frames += 1

    def stop_recording(self):
        if self.writer is None:
            return

        self.writer.close()
        final_name = os.path.splitext(self.filename)[0]
        os.rename(self.filename, final_name)
        self.writer = None
        self.recording = False

    def delete_recording(self):
        if self.writer is None:
            return

        self.writer.close()
        os.remove(self.filename)
        self.writer = None
        self.recording = False


def new_temp_name():
    return datetime.now().strftime("%Y%m%d.%H%M%S.%f" + CPTV_TEMP_EXT)
