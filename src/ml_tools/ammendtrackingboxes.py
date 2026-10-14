import argparse
from ml_tools.logs import init_logging
from pathlib import Path
from ml_tools.rawdb import RawDatabase
import logging
from config.buildconfig import BuildConfig
from ml_tools.dataset import filter_clip, filter_track
from ml_tools.tools import CustomJSONEncoder
import json

VERSION = "0.1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="Weights to load into model")
    parser.add_argument(
        "model",
        help="Path to model file to use, will override config model",
    )
    parser.add_argument(
        "cptv_dir",
        help="Evalute directory of cptv files",
    )
    parser.add_argument("--device", default="0", help="GPU index (default: 0), or cpu")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Images per YOLO predict call"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Processes used to filter and decode cptv files",
    )
    args = parser.parse_args()

    return args


def main():
    init_logging()
    args = parse_args()
    cptv_dir = Path(args.cptv_dir)
    model_path = Path(args.model)

    evaluate_dir(model_path, cptv_dir, args.device, args.batch_size, args.workers)


def evaluate_dir(model_path, dir, device, batch_size, workers):
    logging.info("Evaluating cptv files in %s", dir)
    from multiprocessing import Pool
    import queue
    import itertools

    files = list(dir.glob(f"**/*cptv"))
    files.sort()
    total_files = len(files)
    complete = 0
    # create pool before the model so workers never fork a CUDA context
    with Pool(workers) as pool:
        model = Yolo(model_path, device=device, batch_size=batch_size)
        # bound how many decoded clips are queued, imap_unordered has no
        # backpressure and buffers every result in this process until OOM
        # results are put on done as they finish so we take whichever is first
        done = queue.Queue()

        def on_error(e):
            logging.error("Could not load clip", exc_info=e)
            done.put(None)

        remaining = iter(files)
        in_flight = 0
        for cptv_file in itertools.islice(remaining, workers * 2):
            pool.apply_async(
                load_clip, (cptv_file,), callback=done.put, error_callback=on_error
            )
            in_flight += 1
        while in_flight:
            loaded = done.get()
            in_flight -= 1
            next_file = next(remaining, None)
            if next_file is not None:
                pool.apply_async(
                    load_clip, (next_file,), callback=done.put, error_callback=on_error
                )
                in_flight += 1
            complete += 1
            if complete % 100 == 0:
                logging.info("Done %s / %s ", complete, total_files)
            if loaded is None:
                continue
            cptv_file, metadata_file, metadata, frame_numbers, thermals = loaded
            try:
                frame_boxes = model.predict_batch(thermals)
            except:
                logging.error("Could not predict %s", cptv_file, exc_info=True)
                continue
            yolo_preditions = []
            for frame_number, boxes in zip(frame_numbers, frame_boxes):
                for bb in boxes:
                    bb["frame_number"] = frame_number
                if len(boxes) > 0:
                    yolo_preditions.append(boxes)
            metadata["yolo_bb"] = yolo_preditions
            metadata["yolo_version"] = VERSION
            save_metadata(cptv_file, metadata_file, metadata)
    logging.info("Done %s / %s ", complete, total_files)


def save_metadata(cptv_file, metadata_file, metadata):
    try:
        with metadata_file.open("w") as t:
            # add in some metadata stats
            json.dump(metadata, t, indent=4, cls=CustomJSONEncoder)
    except:
        logging.error("Could not save %s", cptv_file, exc_info=True)


def load_clip(cptv_file):
    """Filter a clip and decode its frames, returns None if it should be skipped"""
    try:
        import numpy as np

        # for clip in dataset.clips:
        reason = {}
        clip_db = RawDatabase(cptv_file)
        # tracks = meta_data.get("Tracks", [])
        metadata = clip_db.meta_data
        if "yolo_bb" in metadata and metadata.get("yolo_version") == VERSION:
            return
        clip = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS)
        if clip is None:
            logging.warn("No clip for %s", cptv_file)
            return None

        if filter_clip(clip, None, None, reason):
            # logging.info("Filtering %s", cptv_file)
            return None
        clip.tracks = [
            track
            for track in clip.tracks
            if not filter_track(track, BuildConfig.EXCLUDED_TAGS, reason)
        ]
        if len(clip.tracks) == 0:
            logging.info("No tracks after filtering %s", cptv_file)
            return None
        clip_db.load_frames()
        if len(clip_db.frames) == 0:
            return None
        frame_numbers = [f.frame_number for f in clip_db.frames]
        # normalise here so we send uint8 frames back to the main process
        thermals = np.stack([grayscale_pixels(f.thermal) for f in clip_db.frames])
        return cptv_file, clip_db.meta_data_file, metadata, frame_numbers, thermals
    except:
        logging.error("Could not load %s", cptv_file, exc_info=True)
    return None


class Yolo:
    TYPE = "YOLO"

    def __init__(
        self,
        model_name,
        run_over_network=False,
        load_model=True,
        device="cpu",
        batch_size=64,
    ):
        # super().__init__(model_name, run_over_network)
        self.model_file = model_name
        import math

        if run_over_network or not load_model:
            return
        self.load_model()
        self.tta = True
        self.imgsz = 320
        self.tiles = False
        self.iou = 0.55
        # fraction of a box inside a larger box to suppress it, 0 disables
        self.overlap = 0.8
        height = 120
        width = 160
        self.device = device
        self.batch_size = batch_size
        sizes = (
            sorted(
                {
                    max(32, round(self.imgsz * scale / 32) * 32)
                    for scale in (0.75, 1, 1.25)
                }
            )
            if self.tta
            else [self.imgsz]
        )
        self.views = [
            (0, 0, width, height, size, flip)
            for size in sizes
            for flip in ([False, True] if self.tta else [False])
        ]
        if self.tiles and min(width, height) >= 8:
            tile_w, tile_h = math.ceil(width * 2 / 3), math.ceil(height * 2 / 3)
            self.views.extend(
                (x, y, x + tile_w, y + tile_h, self.imgsz, False)
                for y in (0, height - tile_h)
                for x in (0, width - tile_w)
            )

    def load_model(self):
        from ultralytics import YOLO

        self.model = YOLO(self.model_file)
        self.labels = self.model.names
        # 3. View the detailed structure and model information

    def predict(self, thermal):
        # if self.run_over_network:
        #     return self.predict_over_network(np.float32(input_x))
        detections, _, _ = self.predict_frame(thermal)
        return detections

    def predict_batch(self, thermals):
        """Predict every frame in thermals, batching each TTA view across frames
        thermals may already be uint8 grayscale from load_clip"""
        import numpy as np

        rgbs = [
            np.repeat(
                (t if t.dtype == np.uint8 else grayscale_pixels(t))[..., None],
                3,
                axis=2,
            )
            for t in thermals
        ]
        height, width = rgbs[0].shape[:2]
        frame_boxes = [[] for _ in rgbs]
        for left, top, right, bottom, size, flip in self.views:
            images = []
            for rgb in rgbs:
                crop = rgb[top:bottom, left:right]
                pixels = crop[:, ::-1] if flip else crop
                # ndarray input to Ultralytics is BGR
                images.append(np.ascontiguousarray(pixels[..., ::-1]))
            for start in range(0, len(images), self.batch_size):
                results = self.model.predict(
                    images[start : start + self.batch_size],
                    imgsz=size,
                    conf=0.1,
                    iou=self.iou if self.tta or self.tiles else 0.7,
                    augment=False,
                    device=self.device,
                    save=False,
                    verbose=False,
                )
                for i, result in enumerate(results):
                    frame_boxes[start + i].extend(
                        self.result_boxes(
                            result, left, top, right, bottom, flip, width, height
                        )
                    )
        for i, boxes in enumerate(frame_boxes):
            if self.tta or self.tiles:
                boxes = merge_detections(boxes, self.iou)
            frame_boxes[i] = suppress_inner(boxes, self.overlap)
        return frame_boxes

    def result_boxes(self, result, left, top, right, bottom, flip, width, height):
        boxes = []
        tile = (left, top, right, bottom) != (0, 0, width, height)
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            x0, y0, x1, y1 = xyxy
            if flip:
                x0, x1 = right - left - x1, right - left - x0
            # Ignore boxes cut by an internal tile edge; other overlapping
            # tiles and the full-frame passes provide their complete view.
            if tile and (
                (left > 0 and x0 <= 1)
                or (top > 0 and y0 <= 1)
                or (right < width and x1 >= right - left - 1)
                or (bottom < height and y1 >= bottom - top - 1)
            ):
                continue
            x0, x1 = max(0, x0 + left), min(width, x1 + left)
            y0, y1 = max(0, y0 + top), min(height, y1 + top)
            if x1 <= x0 or y1 <= y0:
                continue
            class_id = int(class_id)
            boxes.append(
                dict(
                    kind="yolo",
                    label=result.names[class_id],
                    class_id=class_id,
                    confidence=score,
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                )
            )
        return boxes

    def predict_frame(self, thermal):
        gray = grayscale_pixels(thermal)
        import numpy as np
        import math

        rgb = np.repeat(gray[..., None], 3, axis=2)
        height, width = rgb.shape[:2]
        boxes = []

        for left, top, right, bottom, size, flip in self.views:
            crop = rgb[top:bottom, left:right]
            pixels = np.ascontiguousarray(crop[:, ::-1] if flip else crop)
            # ndarray input to Ultralytics is BGR. Thermal channels are identical,
            # but convert explicitly so this is also correct for future RGB images.
            result = self.model.predict(
                np.ascontiguousarray(pixels[..., ::-1]),
                imgsz=size,
                conf=0.1,
                iou=self.iou if self.tta or self.tiles else 0.7,
                augment=False,
                device=self.device,
                save=False,
                verbose=False,
            )[0]
            boxes.extend(
                self.result_boxes(result, left, top, right, bottom, flip, width, height)
            )
        raw_count = len(boxes)
        if self.tta or self.tiles:
            boxes = merge_detections(boxes, self.iou)
        boxes = suppress_inner(boxes, self.overlap)
        return boxes, len(self.views), raw_count

    def shape(self):
        return 1, self.input["shape"]


def merge_detections(boxes, threshold):
    """Class-aware NMS in original-image coordinates; retain actual model scores."""
    kept = []
    for box in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
        suppress = False
        for other in kept:
            if box["class_id"] != other["class_id"]:
                continue
            w = max(
                0,
                min(box["x"] + box["width"], other["x"] + other["width"])
                - max(box["x"], other["x"]),
            )
            h = max(
                0,
                min(box["y"] + box["height"], other["y"] + other["height"])
                - max(box["y"], other["y"]),
            )
            intersection = w * h
            union = (
                box["width"] * box["height"]
                + other["width"] * other["height"]
                - intersection
            )
            if union > 0 and intersection / union > threshold:
                suppress = True
                break
        if not suppress:
            kept.append(box)
    return kept


def suppress_inner(boxes, overlap=0.8):
    """Drop boxes mostly contained in a larger box, regardless of class."""
    if overlap == 0:
        return list(boxes)
    kept = []
    for box in sorted(
        boxes, key=lambda b: (b["width"] * b["height"], b["confidence"]), reverse=True
    ):
        area = box["width"] * box["height"]
        if area <= 0:
            continue
        for outer in kept:
            w = max(
                0,
                min(box["x"] + box["width"], outer["x"] + outer["width"])
                - max(box["x"], outer["x"]),
            )
            h = max(
                0,
                min(box["y"] + box["height"], outer["y"] + outer["height"])
                - max(box["y"], outer["y"]),
            )
            if w * h / area >= overlap:
                break
        else:
            kept.append(box)
    return kept


def grayscale_pixels(pixels):
    import numpy as np

    values = np.asarray(pixels, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.zeros(values.shape, dtype=np.uint8)
    return (
        np.rint((values - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)
    )


if __name__ == "__main__":
    main()
