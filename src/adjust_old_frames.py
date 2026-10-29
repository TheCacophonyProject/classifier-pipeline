"""Fix frame numbers in metadata made by tracker versions < 10.

Before tracker version 10 the cptv background frame was counted as frame 0, so
every frame number in the metadata is one too high, i.e. a track starting at
frame 1 really starts at frame 0. This shifts all frame numbers (and track
start / end times) back by one frame for clips whose cptv has a background frame.

Adjusted metadata gets tracker_version 10, since its frame numbers now follow the
version 10 numbering, which stops rawdb / makedataset adjusting them again. The
old version is kept in original_tracker_version.

For every file:
- only "Tracks" is kept (as audio-training fixtracks.py), "tracks" is removed if
  "Tracks" exists otherwise renamed to "Tracks"
- positions that only have a frameTime get a frame_number instead (same logic as
  rawdb) and frameTime is removed

usage: python3 adjust_old_frames.py <cptv file or folder> (--output-dir <dir> | --in-place)
"""

import argparse
import json
import logging
import os
import sys

from cptv_rs_python_bindings import CptvReader

from ml_tools.tools import CustomJSONEncoder

FRAME_KEYS = ["frame_number", "frameNumber", "order"]


def init_logging():
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=fmt)


def cptv_info(cptv_file):
    """returns (has background frame, fps)"""
    reader = CptvReader(str(cptv_file))
    header = reader.get_header()
    fps = header.fps if header.fps else 9
    frame = reader.next_frame()
    return frame is not None and frame.background_frame, fps


def get_tracker_version(meta):
    version = meta.get("tracker_version")
    if version is None:
        version = meta.get("algorithm", {}).get("tracker_version")
    return version


def fix_tracks(meta):
    """keeps only "Tracks", same as audio-training fixtracks.py, returns the change
    made or None"""
    if "tracks" not in meta:
        return None
    if "Tracks" in meta:
        del meta["tracks"]
        return "removed tracks"
    # rename in place to keep key order
    renamed = {("Tracks" if k == "tracks" else k): v for k, v in meta.items()}
    meta.clear()
    meta.update(renamed)
    return "renamed tracks to Tracks"


def frame_time_to_frame_number(track):
    """replaces frameTime with frame_number using the same logic as rawdb,
    returns True if any positions were changed"""
    changed = False
    prev_frame = None
    for i, position in enumerate(track.get("positions") or []):
        if not isinstance(position, dict):
            continue
        frame = next(
            (position[key] for key in FRAME_KEYS if position.get(key) is not None), None
        )
        if frame is None and "frameTime" in position:
            if i == 0:
                frame = round(position["frameTime"] * 9)
            else:
                frame = prev_frame + 1
            position["frame_number"] = frame
        if "frameTime" in position:
            del position["frameTime"]
            changed = True
        prev_frame = frame
    return changed


def adjust_seconds(track, fps):
    for key in ["start", "end", "start_s", "end_s"]:
        if track.get(key) is not None:
            track[key] = round(max(0, track[key] - 1 / fps), 2)


def adjust_positions(track, clip_id):
    positions = track.get("positions")
    if not positions:
        return
    adjusted = []
    for position in positions:
        if isinstance(position, list):
            # [frame_time, region] format, region may also have a frame number
            position[0] = max(0, position[0] - 1)
            region = position[1]
        else:
            region = position
        dropped = False
        if isinstance(region, dict):
            for key in FRAME_KEYS:
                if region.get(key) is not None:
                    region[key] -= 1
                    dropped = dropped or region[key] < 0
        if dropped:
            # frame 0 was the background frame so no region should be there
            logging.warning(
                "%s track %s has a region on the background frame, dropping it",
                clip_id,
                track.get("id"),
            )
            continue
        adjusted.append(position)
    track["positions"] = adjusted


def adjust_track_frames(track):
    for key in ["frame_start", "frame_end"]:
        if track.get(key) is not None:
            track[key] = max(0, track[key] - 1)

    # predictions written by classify
    for prediction in track.get("predictions") or []:
        for segment in prediction.get("predictions") or []:
            segment["frames"] = [max(0, f - 1) for f in segment.get("frames", [])]
            if segment.get("predicted_at_frame") is not None:
                segment["predicted_at_frame"] = max(
                    0, segment["predicted_at_frame"] - 1
                )

    # old AI tags, used by --reuse-prediction-frames
    for tag in track.get("tags") or []:
        data = tag.get("data")
        if isinstance(data, dict) and data.get("prediction_frames") is not None:
            data["prediction_frames"] = [
                [max(0, f - 1) for f in frames] for frames in data["prediction_frames"]
            ]


def adjust_file(cptv_file, out_file):
    meta_file = os.path.splitext(cptv_file)[0] + ".txt"
    if not os.path.exists(meta_file):
        logging.error("No metadata for %s", cptv_file)
        return False
    with open(meta_file, "r") as f:
        meta = json.load(f)

    clip_id = meta.get("id", cptv_file)
    # api metadata has tracks in both "Tracks" and "tracks", only keep "Tracks"
    tracks_change = fix_tracks(meta)
    if tracks_change:
        logging.info("%s %s", clip_id, tracks_change)
    tracks = meta.get("Tracks") or []

    # frameTime is converted for every file, before any shift as it uses the
    # same frame numbering as the rest of the metadata
    converted = False
    for track in tracks:
        converted = frame_time_to_frame_number(track) or converted
    if converted:
        logging.info("%s converted frameTime to frame_number", clip_id)

    version = get_tracker_version(meta)
    shifted = False
    if meta.get("original_tracker_version") is not None:
        logging.info("%s already adjusted", clip_id)
    elif not isinstance(version, int) or version >= 10:
        logging.info("%s tracker version %s doesn't need adjusting", clip_id, version)
    else:
        has_background, fps = cptv_info(cptv_file)
        if not has_background:
            # bug only shifted frames when there was a background frame
            logging.info("%s has no background frame, not adjusting", clip_id)
        else:
            for track in tracks:
                adjust_seconds(track, fps)
                adjust_positions(track, clip_id)
                adjust_track_frames(track)

            meta["original_tracker_version"] = version
            if "tracker_version" in meta:
                meta["tracker_version"] = 10
            else:
                meta["algorithm"]["tracker_version"] = 10
            shifted = True
            logging.info("%s adjusted from tracker version %s", clip_id, version)

    if not tracks_change and not converted and not shifted:
        return False
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(meta, f, indent=4, cls=CustomJSONEncoder)
    logging.info("%s saved %s", clip_id, out_file)
    return True


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="a cptv file or a folder of cptv files")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument(
        "--output-dir",
        help="write adjusted metadata here, mirroring source subfolders",
    )
    out.add_argument(
        "--in-place", action="store_true", help="overwrite the source metadata"
    )
    args = parser.parse_args()

    if os.path.isfile(args.source):
        base = os.path.dirname(args.source)
        files = [args.source]
    else:
        base = args.source
        files = [
            os.path.join(folder, name)
            for folder, _, names in os.walk(args.source)
            for name in names
            if name.endswith(".cptv")
        ]
    adjusted = 0
    for cptv_file in files:
        meta_file = os.path.splitext(cptv_file)[0] + ".txt"
        if args.in_place:
            out_file = meta_file
        else:
            out_file = os.path.join(args.output_dir, os.path.relpath(meta_file, base))
        try:
            adjusted += adjust_file(cptv_file, out_file)
        except Exception:
            logging.error("Error adjusting %s", cptv_file, exc_info=True)
    logging.info("Adjusted %s of %s files", adjusted, len(files))


if __name__ == "__main__":
    main()
