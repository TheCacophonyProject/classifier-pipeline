"" """
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import logging
import time
from multiprocessing import Process, Queue
import traceback
from ml_tools import tools

from track.clip import Clip, ClipStats
import numpy as np
import json
from pathlib import Path

import h5py
from cptv import CPTVReader
import yaml
from track.cliptrackextractor import is_affected_by_ffc
from track.track import Track
from track.region import Region

FPS = 9


def process_job(loader, queue, out_dir, config):
    i = 0
    while True:
        i += 1
        filename = queue.get()
        logging.info("Processing %s", filename)
        try:
            if filename == "DONE":
                break
            else:
                loader.process_file(str(filename), out_dir, config)
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s %s", filename, e)
            traceback.print_exc()


class ClipLoader:
    def __init__(self, config):
        self.config = config

        # number of threads to use when processing jobs.
        self.workers_threads = 9

    def process_all(self, root, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        job_queue = Queue()
        processes = []
        for i in range(max(1, self.workers_threads)):
            p = Process(
                target=process_job,
                args=(self, job_queue, out_dir, self.config),
            )
            processes.append(p)
            p.start()

        file_paths = []
        #
        for folder_path, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1] in [".cptv"]:
                    full_path = os.path.join(folder_path, name)
                    file_paths.append(full_path)
        # allows us know the order of processing
        file_paths.sort()
        for file_path in file_paths:
            job_queue.put(file_path)

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put("DONE")
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()

    def process_file(self, filename, out_dir, config):
        start = time.time()
        filename = Path(filename)
        logging.info(f"processing %s", filename)
        metadata_file = filename.with_suffix(".txt")
        if not metadata_file.exists():
            logging.error("No meta data found for %s", metadata_file)
            return

        metadata = tools.load_clip_metadata(metadata_file)
        r_id = metadata["id"]
        out_file = out_dir / f"{r_id}.hdf5"
        tracker_version = metadata.get("tracker_version")
        logging.info("Tracker version is %s", tracker_version)
        if out_file.exists() and tracker_version > 9:
            logging.warning("Already loaded %s", filename)
            # going to add some missing fierlds
            with h5py.File(out_file, "a") as f:
                if f.attrs.get("rec_time") is None:
                    with open(filename, "rb") as cptv:
                        reader = CPTVReader(cptv)
                        video_start_time = reader.timestamp.astimezone(Clip.local_tz)
                    f.attrs["rec_time"] = video_start_time.isoformat()
            return
        if len(metadata.get("Tracks")) == 0:
            logging.error("No tracks found for %s", filename)
            return

        clip = Clip(config.tracking["thermal"], filename)
        clip.load_metadata(
            metadata,
            config.load.tag_precedence,
        )

        with h5py.File(out_file, "w") as f:
            try:
                logging.info("creating clip %s", clip.get_id())

                clip_id = str(clip.get_id())

                clip_node = f
                triggered_temp_thresh = None
                camera_model = None
                with open(clip.source_file, "rb") as f:
                    reader = CPTVReader(f)
                    clip.set_res(reader.x_resolution, reader.y_resolution)
                    if clip.from_metadata:
                        for track in clip.tracks:
                            track.crop_regions()
                    if reader.model:
                        camera_model = reader.model.decode()
                    clip.set_model(camera_model)

                    # if we have the triggered motion threshold should use that
                    # maybe even override dynamic threshold with this value
                    if reader.motion_config:
                        motion = yaml.safe_load(reader.motion_config)
                        triggered_temp_thresh = motion.get("triggeredthresh")
                        if triggered_temp_thresh:
                            clip.temp_thresh = triggered_temp_thresh

                    video_start_time = reader.timestamp.astimezone(Clip.local_tz)
                    clip.set_video_stats(video_start_time)
                    print("Adding rec time", video_start_time)
                    frames = clip_node.create_group("frames")
                    ffc_frames = []
                    stats = ClipStats()
                    cropped_stats = ClipStats()
                    num_frames = 0
                    cptv_frames = []
                    region_adjust = 0
                    for frame in reader:
                        if frame.background_frame:
                            back_node = frames.create_dataset(
                                "background",
                                frame.pix.shape,
                                chunks=frame.pix.shape,
                                compression="gzip",
                                dtype=frame.pix.dtype,
                            )
                            back_node[:, :] = frame.pix
                            # pre tracker verison 10 there was a bug where back frame was counted in region frame number
                            #  so frame 0 is actually the background frame, no tracks shouild ever start at 0
                            if tracker_version < 10:
                                region_adjust = -1
                                logging.info("Adjusting regions by %s", region_adjust)
                            continue
                        ffc = is_affected_by_ffc(frame)
                        if ffc:
                            ffc_frames.append(num_frames)

                        cptv_frames.append(frame.pix)
                        stats.add_frame(frame.pix)
                        cropped_stats.add_frame(clip.crop_rectangle.subimage(frame.pix))
                        num_frames += 1
                    cptv_frames = np.uint16(cptv_frames)
                    thermal_node = frames.create_dataset(
                        "thermals",
                        cptv_frames.shape,
                        chunks=(1, *cptv_frames.shape[1:]),
                        compression="gzip",
                        dtype=cptv_frames.dtype,
                    )

                    thermal_node[:, :, :] = cptv_frames
                    stats.completed()
                    cropped_stats.completed()
                    group_attrs = clip_node.attrs
                    group_attrs["clip_id"] = r_id
                    group_attrs["rec_time"] = video_start_time.isoformat()
                    group_attrs["num_frames"] = np.uint16(num_frames)
                    group_attrs["ffc_frames"] = np.uint16(ffc_frames)
                    group_attrs["device_id"] = metadata["deviceId"]
                    stationId = metadata.get("stationId", 0)
                    group_attrs["station_id"] = stationId
                    group_attrs["crop_rectangle"] = np.uint8(
                        clip.crop_rectangle.to_ltrb()
                    )
                    group_attrs["max_temp"] = np.uint16(cropped_stats.max_temp)
                    group_attrs["min_temp"] = np.uint16(cropped_stats.min_temp)
                    group_attrs["mean_temp"] = np.uint16(cropped_stats.mean_temp)
                    group_attrs["frame_temp_min"] = np.uint16(
                        cropped_stats.frame_stats_min
                    )
                    group_attrs["frame_temp_max"] = np.uint16(
                        cropped_stats.frame_stats_max
                    )
                    group_attrs["frame_temp_median"] = np.uint16(
                        cropped_stats.frame_stats_median
                    )
                    group_attrs["frame_temp_mean"] = np.uint16(
                        cropped_stats.frame_stats_mean
                    )
                    group_attrs["start_time"] = clip.video_start_time.isoformat()
                    group_attrs["res_x"] = clip.res_x
                    group_attrs["res_y"] = clip.res_y
                    if camera_model is not None:
                        group_attrs["model"] = clip.camera_model

                    if triggered_temp_thresh is not None:
                        group_attrs["temp_thresh"] = triggered_temp_thresh

                    if clip.tags:
                        clip_tags = []
                        for track in clip.tags:
                            if track["what"]:
                                clip_tags.append(track["what"])
                            elif track["detail"]:
                                clip_tags.append(track["detail"])
                        group_attrs["tags"] = clip_tags

                tracks_group = clip_node.create_group("tracks")

                tracks = metadata.get("Tracks", [])
                for track in tracks:
                    track_id = track["id"]

                    track_group = tracks_group.create_group(str(track_id))

                    node_attrs = track_group.attrs
                    node_attrs["id"] = track_id
                    tags = track.get("tags", [])
                    tag = Track.get_best_human_tag(
                        tags, self.config.load.tag_precedence, 0
                    )

                    master_tag = [
                        t
                        for t in tags
                        if t.get("automatic")
                        and not isinstance(t.get("data", ""), str)
                        and t.get("data", {}).get("name") == "Master"
                    ]
                    if len(master_tag) > 0:
                        master_tag = master_tag[0]
                        node_attrs["ai_tag"] = master_tag["what"]
                        node_attrs["ai_tag_confidence"] = master_tag["confidence"]

                    if tag is not None:
                        node_attrs["human_tag"] = tag["what"]
                        node_attrs["human_tag_confidence"] = tag["confidence"]

                    human_tags = [
                        (t.get("what"), t["confidence"])
                        for t in tags
                        if t.get("automatic", False) != True
                    ]
                    if len(human_tags) > 0:
                        node_attrs["human_tags"] = [h[0] for h in human_tags]
                        node_attrs["human_tags_confidence"] = np.float32(
                            [h[1] for h in human_tags]
                        )

                    start = None
                    end = None

                    prev_frame = None
                    regions = []
                    for i, r in enumerate(track.get("positions")):
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
                        # new_f = region.frame_number + region_adjust
                        prev_frame = region.frame_number
                        region.frame_number = region.frame_number + region_adjust
                        assert region.frame_number >= 0
                        regions.append(region.to_array())
                        if start is None:
                            start = region.frame_number
                        end = region.frame_number
                    node_attrs["start_frame"] = start
                    node_attrs["end_frame"] = min(num_frames, end)

                    region_array = np.uint16(regions)
                    regions = track_group.create_dataset(
                        "regions",
                        region_array.shape,
                        chunks=region_array.shape,
                        compression="gzip",
                        dtype=region_array.dtype,
                    )
                    regions[:, :] = region_array
            except:
                logging.error("Error saving file %s", filename, exc_info=True)
                f.close()
                out_file.unlink()
