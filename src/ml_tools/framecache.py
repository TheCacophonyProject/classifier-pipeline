import h5py
import os
import numpy as np


from ml_tools.frame import Frame, TrackChannels
import logging


class FrameCache:
    def __init__(
        self,
        filename,
        keep_open=True,
        delete_if_exists=True,
        flush_threshold_bytes=10 * 1024 * 1024,
    ):
        self.filename = filename
        self.db = None
        self.keep_open = keep_open
        self.num_frames = 0
        self.flush_threshold_bytes = flush_threshold_bytes
        self.bytes_since_flush = 0
        if delete_if_exists:
            self.delete()

        f = h5py.File(self.filename, "w")
        f.create_group("tracks")
        f.close()

    def add_frame(self, frame, track_id=0):
        self.open()
        tracks = self.db["tracks"]
        if str(track_id) not in tracks:
            logging.info("Adding track id %s", track_id)
            frames = tracks.create_group(str(track_id))
            frames = frames.create_group("frames")
        else:
            frames = tracks[str(track_id)]["frames"]
        frame_group = frames.create_group(str(frame.frame_number))
        frame_group.attrs["ffc_affected"] = frame.ffc_affected
        height, width = frame.thermal.shape

        chunks = (1, height, width)
        channels = []
        dims = 0
        data = []
        if frame.thermal is not None:
            channels.append(TrackChannels.thermal.value)
            dims += 1
            data.append(np.float32(frame.thermal))
        if frame.filtered is not None:
            channels.append(TrackChannels.filtered.value)
            dims += 1
            data.append(np.float32(frame.filtered))

        if frame.flow is not None:
            from ml_tools.tools import get_clipped_flow

            channels.append(TrackChannels.flow.value)
            scaled_flow = get_clipped_flow(frame.flow)
            scaled_flow_h = np.float32(scaled_flow[:, :, 0])
            scaled_flow_v = np.float32(scaled_flow[:, :, 1])
            data.append(scaled_flow_h)
            data.append(scaled_flow_v)
            dims += 2
        if frame.mask is not None:
            channels.append(TrackChannels.mask.value)
            data.append(np.float32(frame.mask))
            dims += 1
        frame_group.attrs["channels"] = np.uint8(channels)

        dims = (dims, height, width)
        frame_node = frame_group.create_dataset(
            "frame", dims, chunks=chunks, dtype=np.float32
        )

        frame_node[:, :, :] = data
        if self.keep_open:
            self.bytes_since_flush += frame_node.size * frame_node.dtype.itemsize
            if self.bytes_since_flush >= self.flush_threshold_bytes:
                self.flush()
        else:
            self.close()

    def get_track_frames(self, track_id):
        self.open()
        frames = self.db["tracks"][str(track_id)]["frames"]
        return frames

    def get_frame(self, frame_number, track_id=0):
        self.open()
        frames = self.db["tracks"][str(track_id)]["frames"]
        frame = self.get_frame_from_group(frames, frame_number)

        if not self.keep_open:
            self.close()
        return frame

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def flush(self):
        if self.db:
            self.db.flush()
            self.bytes_since_flush = 0

    def open(self, mode="a"):
        if not self.db:
            self.db = h5py.File(self.filename, mode)

    def delete(self):
        if self.db:
            self.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)


def get_frame_from_group(raw, frame_number):
    if str(frame_number) in raw:
        frame_group = raw[str(frame_number)]
        frame = frame_group["frame"]
        ffc_affected = frame_group.attrs["ffc_affected"]
        channels = frame_group.attrs["channels"]
        return Frame.from_channels(
            frame,
            channels,
            frame_number,
            flow_clipped=True,
            ffc_affected=ffc_affected,
        )
    else:
        return None
