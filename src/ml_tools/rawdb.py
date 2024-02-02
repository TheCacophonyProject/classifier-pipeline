"""

Author GP

Date 2023


"""

import os
import logging
import filelock
import numpy as np
from dateutil.parser import parse as parse_date
from .frame import Frame, TrackChannels
import json
from pathlib import Path
import numpy as np
from track.region import Region
from ml_tools.datasetstructures import TrackHeader, ClipHeader
from track.track import Track
from track.cliptrackextractor import is_affected_by_ffc
from cptv import CPTVReader
from ml_tools.tools import Rectangle

special_datasets = [
    "tag_frames",
    "original_frames",
    "background_frame",
    "predictions",
    "overlay",
]


class RawDatabase:
    def __init__(self, database_filename):
        self.file = Path(database_filename)
        self.meta_data_file = self.file.with_suffix(".txt")
        self._meta_data = None
        self.background = None
        self.ffc_frames = None
        self.frames = None
        self.crop_rectangle = Rectangle(1, 1, 160 - 2, 140 - 2)

    def frames_kept(self):
        return None

    def get_frame(self, frame_number):
        if self.frames is None or frame_number > len(self.frames):
            return None
        return self.frames[frame_number]

    def get_frames(self):
        return self.frames

    def get_clip_background(self):
        return self.background

    def load_frames(self):
        ffc_frames = []
        cptv_frames = []
        background = None
        tracker_version = self.meta_data.get("tracker_version")
        frame_i = 0
        with open(self.file, "rb") as f:
            reader = CPTVReader(f)
            for frame in reader:
                if frame.background_frame:
                    background = frame.pix
                    # bug in previous tracker version where background was first frame
                    if tracker_version >= 10:
                        continue
                ffc = is_affected_by_ffc(frame)
                if ffc:
                    ffc_frames.append(frame_i)
                cptv_frames.append(frame.pix)
                frame_i += 1
        frames = np.uint16(cptv_frames)
        if background is None:
            background = np.mean(frames, axis=0)

        self.frames = []
        for i, f in enumerate(frames):
            self.frames.append(Frame(f, None, None, i))
        self.ffc_frames = ffc_frames
        self.background = background

    @property
    def meta_data(self):
        if self._meta_data is not None:
            return self._meta_data
        if not self.meta_data_file.is_file():
            logging.warn("Could not load meta data for %s", self.meta_data_file)
            return None
        with open(self.meta_data_file, "r") as t:
            # add in some metadata stats
            self._meta_data = json.load(t)
        return self._meta_data

    def get_clip_tracks(self, tag_precedence):
        metadata = self.meta_data
        if metadata is None:
            return None
        edge_pixels = metadata.get("edgePixels", 1)
        resx = metadata.get("resX", 160)
        resy = metadata.get("resY", 140)

        self.crop_rectangle = Rectangle(
            edge_pixels, edge_pixels, resx - edge_pixels * 2, resy - edge_pixels * 2
        )

        clip_header = ClipHeader(
            clip_id=int(metadata["id"]),
            station_id=metadata.get("stationId"),
            source_file=self.file,
            location=metadata.get("location"),
            camera=metadata.get("deviceId"),
            rec_time=parse_date(metadata["recordingDateTime"]),
            frames_per_second=10 if self.file.suffix == "mp4" else 9,
            events=metadata.get("event", ""),
            trap=metadata.get("trap", ""),
            tracks=[],
            ffc_frames=self.ffc_frames,
        )
        tracks = metadata.get("Tracks", [])
        meta = []
        for track_meta in tracks:
            tags = track_meta.get("tags", [])
            tag = Track.get_best_human_tag(tags, tag_precedence, 0)
            human_tag = None
            human_tag_confidence = None
            master_tag = [
                t
                for t in tags
                if t.get("automatic")
                and not isinstance(t.get("data", ""), str)
                and t.get("data", {}).get("name") == "Master"
            ]
            if len(master_tag) > 0:
                master_tag = master_tag[0]
                track_meta["ai_tag"] = master_tag["what"]
                track_meta["ai_tag_confidence"] = master_tag["confidence"]

            if tag is not None:
                human_tag = tag["what"]
                human_tag_confidence = tag["confidence"]
            human_tags = [
                (t.get("what"), t["confidence"])
                for t in tags
                if t.get("automatic", False) != True
            ]

            start = None
            end = None

            prev_frame = None
            regions = {}
            for i, r in enumerate(track_meta.get("positions")):
                if isinstance(r, list):
                    region = Region.region_from_array(r[1])
                    if region.frame_number is None:
                        if i == 0:
                            frame_number = round(r[0] * FPS)
                            region.frame_number = frame_number
                        else:
                            region.frame_number = prev_frame + 1
                else:
                    region = Region.region_from_json(r)
                if region.frame_number is None:
                    if "frameTime" in r:
                        if i == 0:
                            region.frame_number = round(r["frameTime"] * 9)
                        else:
                            region.frame_number = prev_frame + 1
                prev_frame = region.frame_number
                region.frame_number = region.frame_number
                assert region.frame_number >= 0
                regions[region.frame_number] = region
                if start is None:
                    start = region.frame_number
                end = region.frame_number
            header = TrackHeader(
                clip_id=clip_header.clip_id,
                track_id=int(track_meta["id"]),
                label=human_tag,
                num_frames=len(regions),
                regions=regions,
                start_frame=start,
                confidence=human_tag_confidence,
                human_tags=human_tags,
                source_file=self.file,
                mega_missed_regions=track_meta.get("mega_missed_regions"),
                # frame_temp_median=frame_temp_median,
            )
            clip_header.tracks.append(header)
        return clip_header

    def get_clip_meta(self, tag_precedence):
        return self.get_clip_tracks(tag_precedence)
        #
        # clip_meta = {}
        # clip_meta["clip_id"] = r_id
        # stationId = metadata.get("stationId", 0)
        # clip_meta["station_id"] = stationId
        # clip_meta["crop_rectangle"] = np.uint8(clip.crop_rectangle.to_ltrb())
