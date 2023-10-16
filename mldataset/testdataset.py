"""
Processes a CPTV file identifying and tracking regions of interest, and saving them in the 'trk' format.
"""

import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import h5py
import cv2
import numpy as np


def parse_params():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "target",
        default=None,
        help="h5py file to display",
    )
    args = parser.parse_args()
    args.target = Path(args.target)
    return args


Track = namedtuple("Track", "id regions what ai_what")


RESCALE = 4
from dateutil.parser import parse


def latest_date(dir):
    import json

    latest_date = None
    dbFiles = dir.glob("**/*.cptv")
    for dbName in dbFiles:
        meta_f = dbName.with_suffix(".txt")
        with open(meta_f, "r") as f:
            # add in some metadata stats
            meta = json.load(f)
        if meta.get("recordingDateTime"):
            meta["recordingDateTime"] = parse(meta["recordingDateTime"])
            if latest_date is None or meta["recordingDateTime"] > latest_date:
                latest_date = meta["recordingDateTime"]
    print("Latest date is", latest_date)


def makecsv(dir):
    dbFiles = dir.glob("**/*.hdf5")
    import csv

    print("Making csv")

    rec_counts = {}
    track_counts = {}
    with open("file-desc.csv", "w") as csvout:
        writer = csv.writer(csvout)
        writer.writerow(["file", "track id", "human tags", "extra human tags"])
        for dbName in dbFiles:
            f = h5py.File(dbName, "r")
            tracks = f["tracks"]
            track_ids = tracks.keys()
            has_track = False
            for track_id in track_ids:
                has_track = True
                human_tags = set()
                extra_tags = set()
                attrs = tracks[track_id].attrs
                if "human_tag" not in attrs:
                    continue
                tag = attrs["human_tag"]
                human_tags.add(tag)
                if "human_tags" in attrs:
                    extra = attrs["human_tags"]
                    for e in extra:
                        extra_tags.add(e)
                if tag in track_counts:
                    track_counts[tag] += 1
                else:
                    track_counts[tag] = 1
                writer.writerow([dbName, track_id, human_tags, extra_tags])
            if not has_track:
                writer.writerow([dbName, None, None, None])
            human_tags = list(human_tags)
            human_tags.sort()
            for t in human_tags:
                if t in rec_counts:
                    rec_counts[t] += 1
                else:
                    rec_counts[t] = 1

    print("REC Counts", rec_counts)
    print("Track Counts", track_counts)


def main():
    args = parse_params()
    latest_date(args.target)
    # makecsv(args.target)
    return
    f = h5py.File(args.target, "r")
    clip_attrs = f.attrs
    print(
        "Camera model is",
        clip_attrs.get("model", ""),
        " res:",
        clip_attrs["res_x"],
        clip_attrs["res_y"],
        " id: ",
        clip_attrs["clip_id"],
        " station id: ",
        clip_attrs["station_id"],
        " tags: ",
        clip_attrs.get("tags"),
    )
    for k in clip_attrs.keys():
        print(f"{k}: {clip_attrs[k]}")
    ffc_frames = clip_attrs["ffc_frames"]
    frames = f["frames"]
    if "background" in frames:
        background = frames["background"]
    else:
        print("No background found so using first frame")
        background = frames["thermals"][0]
    crop_rectangle = clip_attrs["crop_rectangle"]

    background = background[
        crop_rectangle[1] : crop_rectangle[3], crop_rectangle[0] : crop_rectangle[2]
    ]
    norm_back = norm_image(background)

    height, width, _ = norm_back.shape
    norm_back = cv2.resize(norm_back, (width * RESCALE, height * RESCALE))
    norm_back = cv2.putText(
        norm_back,
        "Background",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("b", norm_back)
    cv2.waitKey()
    cv2.destroyAllWindows()

    num_frames = clip_attrs["num_frames"]

    tracks_node = f["tracks"]
    tracks = []
    track_ids = tracks_node.keys()
    for track_id in track_ids:
        track_node = tracks_node[track_id]
        positions = track_node["regions"]
        id = track_node.attrs["id"]
        attrs = track_node.attrs
        human_tag = ""
        ai_tag = ""
        if "human_tag" in attrs:
            tag = attrs["human_tag"]
            conf = attrs["human_tag_confidence"]
            human_tag = f"{tag}-{conf}"

        if "ai_tag" in attrs:
            tag = attrs["ai_tag"]
            conf = attrs["ai_tag_confidence"]
            ai_tag = f"{tag}-{conf}"
        track = Track(id, positions, human_tag, ai_tag)
        tracks.append(track)
        print("For track", track_id)
        for k in attrs.keys():
            print(f"{k}: {attrs[k]}")
    frame_i = 0
    for frame in frames["thermals"]:
        frame = frame[
            crop_rectangle[1] : crop_rectangle[3], crop_rectangle[0] : crop_rectangle[2]
        ]

        filtered = np.float32(frame) - background
        filtered[filtered < 0] = 0
        filtered = norm_image(filtered)
        frame = norm_image(frame)

        height, width, _ = frame.shape
        frame = cv2.resize(frame, (width * RESCALE, height * RESCALE))
        filtered = cv2.resize(filtered, (width * RESCALE, height * RESCALE))

        if frame_i in ffc_frames:
            frame = cv2.putText(
                frame,
                "FFC",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            filtered = cv2.putText(
                filtered,
                "FFC",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        wait = 100
        for t in tracks:
            for r in t.regions:
                if r[4] == frame_i:
                    frame = cv2.rectangle(
                        frame,
                        (r[0] * 4, r[1] * 4),
                        (r[2] * 4, r[3] * 4),
                        (0, 255, 0),
                        1,
                    )
                    frame = cv2.putText(
                        frame,
                        f"human {t.what} \nai {t.ai_what}",
                        (r[0] * 4, r[1] * 4 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    filtered = cv2.rectangle(
                        filtered,
                        (r[0] * 4, r[1] * 4),
                        (r[2] * 4, r[3] * 4),
                        (0, 255, 0),
                        1,
                    )
                    filtered = cv2.putText(
                        filtered,
                        f"human {t.what} \nai {t.ai_what}",
                        (r[0] * 4, r[1] * 4 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    wait = 0
        cv2.imshow("thermal", frame)
        cv2.imshow("filtered", filtered)
        cv2.moveWindow("thermal", 0, 0)
        cv2.moveWindow("filtered", width * 5, 0)
        cv2.waitKey(wait)
        frame_i += 1
    print("DONE")
    cv2.waitKey(0)


def norm_image(image):
    max = np.amax(image)
    min = np.amin(image)
    norm = np.float32(image)
    norm = 255 * (norm - min) / (max - min)

    norm = np.uint8(norm)
    norm = norm[:, :, np.newaxis]
    norm = np.repeat(norm, 3, axis=2)
    return norm


if __name__ == "__main__":
    main()
