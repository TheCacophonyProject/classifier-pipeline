from abc import ABC, abstractmethod

import json
import logging

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
        preprocessed, mass = self.preprocess(track, args)
        if preprocessed is None:
            return None, None
        pred = self.predict(np.array(preprocessed))
        return pred, mass

    def preprocess(self, clip, track, **args):
        if self.model_type == "RandomForest":
            return
        last_x_frames = args.get("last_x_frames", 1)
        scale = args.get("scale", None)
        if self.type == "IR":
            logging.info("Preprocess IR scale %s last_x %s", scale, last_x_frames)
            from ml_tools.preprocess import (
                preprocess_ir,
            )

            regions = track.bounds_history[-last_x_frames:]
            frames = clip.frame_buffer.get_last_x(len(regions))
            preprocessed = []
            masses = []
            for i in range(len(regions)):
                regions[i] = regions[i].copy()
                if scale and not region.blank:
                    regions[i].rescale(1 / scale)

                region = regions[i]

                frame = frames[i]

                if (
                    frame is None
                    or region.width == 0
                    or region.height == 0
                    or region.blank
                ):
                    continue
                params = self.params
                preprocessed = preprocess_ir(
                    frame.copy(),
                    (
                        params.frame_size,
                        params.frame_size,
                    ),
                    region=region,
                    preprocess_fn=self.preprocess_fn,
                )
                if preprocessed is None:
                    continue
                preprocessed.append(preprocessed)
                masses.append(1)
            return preprocessed, masses
        elif self.type == "thermal":
            from ml_tools.preprocess import (
                preprocess_movement,
            )

            frames_per_classify = args.get("frames_per_classify", 25)
            logging.info("Preprocess IR scale %s last_x %s", scale, last_x_frames)

            regions = track.bounds_history[-last_x_frames:]
            frames = clip.frame_buffer.get_last_x(len(regions))
            if frames is None:
                return
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
            return [preprocessed], [mass]
