#!/usr/bin/python3
import h5py
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime
from PIL import Image

from ml_tools.mpeg_creator import MPEGCreator
from ml_tools import tools


class ClipSaver:
    def __init__(self, name, keep_open=True, delete_if_exists=True):

        file = name + datetime.now().strftime("%Y-%m-%d-%H%M%S") + ".h5py"
        self.filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        self.db = None
        self.keep_open = keep_open

        if delete_if_exists:
            self.delete()

        if os.path.isfile(self.filename) is False:
            f = h5py.File(self.filename, "w")
            f.create_group("clips")
            f.close()

    def add_clip(self, clip):
        self.open()

        clips = self.db["clips"]
        clip_node = clips.create_group(clip.get_id())
        group_attrs = clip_node.attrs
        group_attrs["start_time"] = clip.video_start_time.isoformat()
        group_attrs["threshold"] = clip.threshold
        group_attrs["num_frames"] = clip.frame_on
        group_attrs["threshold"] = clip.temp_thresh
        group_attrs["preview_frames"] = clip.num_preview_frames

        for track in clip.tracks:
            self.add_track(clip, clip_node, track)

        frames_node = clip_node.create_group("frames")
        for frame in clip.frame_buffer.frames:

            height, width = frame.thermal.shape

            chunks = (height, width)

            dims = (height, width)
            frame_node = frames_node.create_dataset(
                str(frame.frame_number), dims, chunks=chunks, dtype=np.uint16
            )
            therm = frame.thermal
            if len(therm[therm > 10000]):
                print(
                    "thermal frame max {} thermal frame min {} frame {}".format(
                        np.amax(therm), np.amin(therm), frame.frame_number
                    )
                )
            frame_node[:, :] = frame.thermal
        clip_node.attrs["finished"] = True

        self.db.flush()
        if not self.keep_open:
            self.close()

    def add_track(self, clip, clip_node, track):
        track_id = str(track.get_id())
        track_node = clip_node.create_group(track_id)
        node_attrs = track_node.attrs
        # write out attributes
        start_time, end_time = clip.start_and_end_in_secs(track)
        node_attrs["id"] = track_id
        node_attrs["frames"] = track.frames
        node_attrs["start_frame"] = track.start_frame
        node_attrs["end_frame"] = track.end_frame
        if track.avg_novelty:
            node_attrs["avg_novelty"] = track.avg_novelty
        if track.tag:
            node_attrs["tag"] = track.tag
        if track.max_novelty:
            node_attrs["max_novelty"] = track.max_novelty
        if track.confidence:
            node_attrs["confidence"] = track.confidence
        if start_time:
            node_attrs["start_time_s"] = start_time
        if end_time:
            node_attrs["end_time_s"] = end_time

        self.db.flush()

        # mark the record as have been writen to.
        # this means if we are interupted part way through the track will be overwritten
        clip_node.attrs["finished"] = True

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


def saveall_to_mp4(db_name, filename):
    db = h5py.File(db_name, mode="r")
    clips = db["clips"]
    for clip_id in clips:
        print("saving {}".format(clip_id))
        clip_to_mp4(db_name, clip_id, filename, db=db)


def clip_to_mp4(db_name, clip_id, filename, db=None):
    if db is None:
        db = h5py.File(db_name, mode="r")
    clips = db["clips"]

    if str(clip_id) in clips:
        clip = clips[str(clip_id)]
        frames = clip["frames"]
        thermals = []
        frame_ids = []
        for frame in frames:
            frame_ids.append(int(frame))
        frame_ids.sort()
        for frame_id in frame_ids:
            thermals.append(np.uint16(frames[str(frame_id)]))
        frames_to_mp4(thermals, clip_id, filename)


def frames_to_mp4(thermals, clip_id, filename):
    mpeg = MPEGCreator(filename + str(clip_id) + ".mp4")
    thermals = np.uint16(thermals)
    t_min = np.amin(thermals)
    t_max = np.amax(thermals)
    for thermal in thermals:

        image = convert_and_resize(thermal, t_min, t_max, 4)
        mpeg.next_frame(np.asarray(image))
    mpeg.close()


def convert_and_resize(frame, h_min, h_max, size=None, mode=Image.BILINEAR):
    """ Converts the image to colour using colour map and resize """
    thermal = frame[:120, :160].copy()
    image = tools.convert_heat_to_img(frame, None, h_min, h_max)
    if size:
        image = image.resize((int(image.width * size), int(image.height * size)), mode)
    return image
