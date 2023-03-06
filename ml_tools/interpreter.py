from abc import ABC, abstractmethod

import json
import logging
import numpy as np
from ml_tools.hyperparams import HyperParams
from pathlib import Path


class Interpreter(ABC):
    def __init__(self, model_file, data_type):
        self.load_json(model_file)
        self.data_type = data_type

    def load_json(self, filename):
        """Loads model and parameters from file."""
        filename = Path(filename)
        filename = filename.with_suffix(".txt")
        logging.info("Loading metadata from %s", filename)
        stats = json.load(open(filename, "r"))

        self.labels = stats["labels"]
        self.params = HyperParams()
        self.params.update(stats.get("hyperparams", {}))

    @abstractmethod
    def shape(self):
        """Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def predict_track(self, clip, track, **args):
        frames, preprocessed, mass = self.preprocess(clip, track, args)
        # print("preprocess is %s", preprocessed)
        if preprocessed is None or len(preprocessed) == 0:
            return None, None, None
        pred = self.predict(np.array(preprocessed))
        return frames, pred, mass

    def preprocess(self, clip, track, args):
        if self.TYPE == "RandomForest":
            return
        last_x_frames = args.get("last_x_frames", 1)
        scale = args.get("scale", None)
        if self.data_type == "IR":
            logging.info("Preprocess IR scale %s last_x %s", scale, last_x_frames)
            from ml_tools.preprocess import (
                preprocess_ir,
            )

            frame_ago = 1
            # get non blank frames
            regions = []
            frames = []
            for r in reversed(track.bounds_history):
                if not r.blank:
                    frame = clip.frame_buffer.get_frame_ago(frame_ago)
                    if frame is None:
                        break
                    frames.append(frame)
                    regions.append(r)
                    if len(regions) == last_x_frames:
                        break
                frame_ago += 1
            if len(frames) == 0:
                return None, None
            preprocessed = []
            masses = []
            for region, frame in zip(regions, frames):
                if (
                    frame is None
                    or region.width == 0
                    or region.height == 0
                    or region.blank
                ):
                    continue
                params = self.params

                pre_f = preprocess_ir(
                    frame.copy(),
                    (
                        params.frame_size,
                        params.frame_size,
                    ),
                    region=region,
                    preprocess_fn=self.preprocess_fn,
                )
                if pre_f is None:
                    continue
                preprocessed.append(pre_f)
                masses.append(1)
            return [frame.frame_number for f in frames], preprocessed, masses
        elif self.data_type == "thermal":
            from ml_tools.preprocess import (
                preprocess_movement,
            )

            frames_per_classify = args.get("frames_per_classify", 25)
            logging.info(
                "Preprocess thermal scale %s frames_per_classify %s last_x %s",
                scale,
                frames_per_classify,
                last_x_frames,
            )
            frame_ago = 0
            # get non blank frames
            regions = []
            frames = []
            for r in reversed(track.bounds_history):
                if not r.blank:
                    frame = clip.frame_buffer.get_frame_ago(frame_ago)
                    if frame is None:
                        break
                    frame = frame
                    regions.append(r)
                    frames.append(frame)
                    assert frame.frame_number == r.frame_number
                    if len(regions) == last_x_frames:
                        break
                frame_ago += 1
            if len(frames) == 0:
                return None, None
            indices = np.random.choice(
                len(regions),
                min(frames_per_classify, len(regions)),
                replace=False,
            )
            indices.sort()
            frames = np.array(frames)[indices]
            regions = np.array(regions)[indices]

            refs = []
            segment_data = []
            mass = 0
            params = self.params

            for frame, region in zip(frames, regions):
                if region.blank:
                    continue
                refs.append(np.median(frame.thermal))
                thermal_reference = np.median(frame.thermal)
                f = frame.crop_by_region(region)
                mass += region.mass
                f.resize_with_aspect(
                    (params.frame_size, params.frame_size),
                    clip.crop_rectangle,
                    True,
                )
                segment_data.append(f)

            preprocessed = preprocess_movement(
                segment_data,
                params.square_width,
                params.frame_size,
                red_type=params.red_type,
                green_type=params.green_type,
                blue_type=params.blue_type,
                preprocess_fn=self.preprocess_fn,
                reference_level=refs,
                keep_edge=params.keep_edge,
            )
            if preprocessed is None:
                return None, mass
            return [f.frame_number for f in frames], [preprocessed], [mass]
