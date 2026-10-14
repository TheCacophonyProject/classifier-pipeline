from ml_tools.rawdb import RawDatabase
from ml_tools.ammendtrackingboxes import grayscale_pixels, save_metadata
from config.buildconfig import BuildConfig
from pathlib import Path
import argparse
import cv2
import numpy as np

VERSION = "0.1"
# BGR
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


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
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.2,
        help="Only preview yolo boxes and yolo_positions with confidence above this",
    )
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Merge overlapping yolo boxes and save those overlapping each track to the track's yolo_positions in the metadata",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Make a video with yolo boxes (yellow), tracks (blue) and yolo_positions (red)",
    )
    args = parser.parse_args()
    if not args.write_meta and not args.preview:
        parser.error("Nothing to do, specify --write-meta and/or --preview")

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


def overlaps(region, bb):
    return (
        region.x < bb["x"] + bb["width"]
        and bb["x"] < region.x + region.width
        and region.y < bb["y"] + bb["height"]
        and bb["y"] < region.y + region.height
    )


def boxes_overlap(a, b):
    return (
        a["x"] < b["x"] + b["width"]
        and b["x"] < a["x"] + a["width"]
        and a["y"] < b["y"] + b["height"]
        and b["y"] < a["y"] + a["height"]
    )


def merge_yolo_boxes(boxes):
    # repeatedly union any overlapping boxes, keeping the most confident label
    boxes = [dict(bb) for bb in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if not boxes_overlap(a, b):
                    continue
                x0 = min(a["x"], b["x"])
                y0 = min(a["y"], b["y"])
                x1 = max(a["x"] + a["width"], b["x"] + b["width"])
                y1 = max(a["y"] + a["height"], b["y"] + b["height"])
                best = a if a.get("confidence", 0) >= b.get("confidence", 0) else b
                boxes[i] = {
                    **best,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
                del boxes[j]
                merged = True
                break
            if merged:
                break
    return boxes


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


def get_yolo_positions(tracks, yolo_boxes):
    # for each track frame, the most confident merged yolo box overlapping it
    yolo_positions = {}
    for track in tracks:
        positions = []
        for frame_number, region in sorted(track.regions_by_frame.items()):
            frame_yolo = [
                bb
                for bb in merge_yolo_boxes(yolo_boxes.get(frame_number, []))
                if overlaps(region, bb)
            ]
            if frame_yolo:
                best = max(frame_yolo, key=lambda bb: bb.get("confidence", 0))
                positions.append({**best, "frame_number": frame_number})
        yolo_positions[track.track_id] = positions
    return yolo_positions


def write_meta(clip_db, tracks, yolo_boxes):
    yolo_positions = get_yolo_positions(tracks, yolo_boxes)
    metadata = clip_db.meta_data
    track_meta = metadata.get("Tracks") or metadata.get("tracks", [])
    for t in track_meta:
        t["yolo_positions"] = yolo_positions.get(int(t["id"]), [])
    save_metadata(clip_db.file, clip_db.meta_data_file, metadata)
    print("Saved yolo positions to", clip_db.meta_data_file)


def preview(clip_db, tracks, yolo_boxes, args):
    track_meta = clip_db.meta_data.get("Tracks") or clip_db.meta_data.get("tracks", [])
    yolo_positions = {}
    for t in track_meta:
        for bb in t.get("yolo_positions", []):
            yolo_positions.setdefault(bb["frame_number"], []).append((t["id"], bb))

    output = args.output
    if output is None:
        output = Path(args.cptv_file).with_suffix(".mp4")
    scale = args.scale
    clip_db.load_frames()
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
        for bb in yolo_boxes.get(f.frame_number, []):
            if bb.get("confidence", 0) <= args.confidence:
                continue
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
                BLUE,
                scale,
                f"{track.track_id} {track.label}",
                below=True,
            )
        # drawn last so they sit on top
        for track_id, bb in yolo_positions.get(f.frame_number, []):
            if bb.get("confidence", 0) <= args.confidence:
                continue
            draw_box(
                img,
                bb["x"],
                bb["y"],
                bb["width"],
                bb["height"],
                RED,
                scale,
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


def main():
    args = parse_args()
    clip_db = RawDatabase(args.cptv_file)
    tracks = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS).tracks
    yolo_boxes = yolo_by_frame(clip_db.meta_data)
    if args.write_meta:
        write_meta(clip_db, tracks, yolo_boxes)
    if args.preview:
        preview(clip_db, tracks, yolo_boxes, args)


if __name__ == "__main__":
    main()
