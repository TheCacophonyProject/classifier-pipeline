from ml_tools.rawdb import RawDatabase
from ml_tools.ammendtrackingboxes import grayscale_pixels
from config.buildconfig import BuildConfig
from pathlib import Path
import argparse
import cv2
import numpy as np

VERSION = "0.1"
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cptv_file",
        help="File to eval",
    )
    parser.add_argument("-o", "--output", help="Output mp4 file")
    parser.add_argument(
        "-s", "--scale", type=int, default=4, help="Upscale factor for video"
    )
    parser.add_argument("--fps", type=int, default=9, help="Frames per second")
    args = parser.parse_args()

    return args


def yolo_by_frame(metadata):
    # yolo_bb is a list of per frame lists of boxes
    by_frame = {}
    for boxes in metadata.get("yolo_bb", []):
        if isinstance(boxes, dict):
            boxes = [boxes]
        for bb in boxes:
            by_frame.setdefault(bb["frame_number"], []).append(bb)
    return by_frame


def draw_box(img, x, y, width, height, colour, scale, text=None, below=False):
    x0, y0 = int(round(x * scale)), int(round(y * scale))
    x1, y1 = int(round((x + width) * scale)), int(round((y + height) * scale))
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, 1)
    if text is not None:
        cv2.putText(
            img,
            text,
            (x0, min(img.shape[0] - 2, y1 + 10)) if below else (x0, max(10, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()
    clip_db = RawDatabase(args.cptv_file)
    clip_db.load_frames()
    tracks = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS).tracks
    metadata = clip_db.meta_data
    yolo_boxes = yolo_by_frame(metadata)

    output = args.output
    if output is None:
        output = Path(args.cptv_file).with_suffix(".mp4")
    scale = args.scale
    frames = clip_db.get_frames()
    height, width = frames[0].thermal.shape
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width * scale, height * scale),
    )
    for f in frames:
        gray = grayscale_pixels(f.thermal)
        img = cv2.cvtColor(np.uint8(gray), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(
            img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
        )
        for track in tracks:
            region = track.regions_by_frame.get(f.frame_number)
            if region is None:
                continue
            draw_box(
                img,
                region.x,
                region.y,
                region.width,
                region.height,
                GREEN,
                scale,
                f"{track.track_id} {track.label}",
                below=True,
            )
        for bb in yolo_boxes.get(f.frame_number, []):
            draw_box(
                img,
                bb["x"],
                bb["y"],
                bb["width"],
                bb["height"],
                YELLOW,
                scale,
                f"{bb.get('label', '')} {bb.get('confidence', 0):.2f}",
            )
        cv2.putText(
            img,
            str(f.frame_number),
            (2, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()
    print("Saved", output)


if __name__ == "__main__":
    main()
