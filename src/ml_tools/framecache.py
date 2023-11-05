import h5py
import os
import numpy as np


from ml_tools.tools import get_clipped_flow
from ml_tools.frame import Frame, TrackChannels


class FrameCache:
    def __init__(self, cptv_name, keep_open=True, delete_if_exists=True):
        basename = os.path.splitext(cptv_name)[0]
        self.filename = basename + ".cache"
        self.db = None
        self.keep_open = keep_open
        self.num_farmes = 0
        if delete_if_exists:
            self.delete()

        f = h5py.File(self.filename, "w")
        f.create_group("frames")
        f.close()

    def add_frame(self, frame):
        self.open()
        frames = self.db["frames"]
        frame_group = frames.create_group(str(frame.frame_number))
        frame_group.attrs["ffc_affected"] = frame.ffc_affected
        height, width = frame.thermal.shape

        chunks = (1, height, width)
        channels = []
        dims = 0
        data = []
        if frame.thermal is not None:
            channels.append(TrackChannels.thermal)
            dims += 1
            data.append(np.float32(frame.thermal))
        if frame.filtered is not None:
            channels.append(TrackChannels.filtered)
            dims += 1
            data.append(np.float32(frame.filtered))

        if frame.flow is not None:
            channels.append(TrackChannels.flow)
            scaled_flow = get_clipped_flow(frame.flow)
            scaled_flow_h = np.float32(scaled_flow[:, :, 0])
            scaled_flow_v = np.float32(scaled_flow[:, :, 1])
            data.append(scaled_flow_h)
            data.append(scaled_flow_v)
            dims += 2
        if frame.mask is not None:
            channels.append(TrackChannels.mask)
            data.append(np.float32(frame.mask))
            dims += 1
        frame_group.attrs["channels"] = np.uint8(channels)

        dims = (dims, height, width)
        frame_node = frame_group.create_dataset(
            "frame", dims, chunks=chunks, dtype=np.float32
        )

        frame_node[:, :, :] = data
        if not self.keep_open:
            self.close()

    def get_frame(self, frame_number):
        self.open()
        frame = None
        if str(frame_number) in self.db["frames"]:
            frame_group = self.db["frames"][str(frame_number)]
            frame = frame_group["frame"]
            ffc_affected = frame_group.attrs["ffc_affected"]
            channels = frame_group.attrs["channels"]
            frame = Frame.from_channels(
                frame,
                channels,
                frame_number,
                flow_clipped=True,
                ffc_affected=ffc_affected,
            )

        if not self.keep_open:
            self.close()
        return frame

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def open(self, mode="a"):
        if not self.db:
            self.db = h5py.File(self.filename, mode)

    def delete(self):
        if self.db:
            self.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)
