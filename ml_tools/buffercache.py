import h5py
import os
import numpy as np
from multiprocessing import Lock


class BufferCache:
    def __init__(self, cptv_name, keep_open=True, delete_if_exists=True):
        basename = os.path.splitext(os.path.basename(cptv_name))[0]
        self.filename = "cache-tmp/"+basename + ".tmp"
        self.db = None
        self.keep_open = keep_open
        
        if delete_if_exists:
            self.delete()

        if not os.path.exists("cache-tmp"):
             os.makedirs("cache-tmp")
        f = h5py.File(self.filename, "w")
        f.create_group("frames")
        f.close()

    def add_frame(self, frame):
        self.open()
        frames = self.db["frames"]
        height, width = frame.thermal.shape

        chunks = (1, height, width)

        dims = (5, height, width)
        frame_node = frames.create_dataset(
            str(frame.frame_number), dims, chunks=chunks, dtype=np.float16
        )
        scaled_flow = np.clip(frame.flow * 256, -16000, 16000)
        frame_val = (
            np.float16(frame.thermal),
            np.float16(frame.filtered),
            np.float16(scaled_flow[:, :, 0]),
            np.float16(scaled_flow[:, :, 1]),
            np.float16(frame.mask),
        )
        frame_node[:, :, :] = frame_val
        if not self.keep_open:
            self.close()

    def get_frame(self, frame_number):
        if not self.db:
            self.open()
        frame = self.db["frames"][str(frame_number)]
        if not self.keep_open:
            self.close()
        return frame

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def open(self, mode = "a"):
        if not self.db:
            self.db = h5py.File(self.filename, mode)

    def delete(self):
        if self.db:
            self.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)
