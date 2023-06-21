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


def main():
    args = parse_params()
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
    ffc_frames = clip_attrs["ffc_frames"]
    frames = f["frames"]
    background = frames["background"]
    crop_rectangle = clip_attrs["crop_rectangle"]

    background = background[
        crop_rectangle[1] : crop_rectangle[3], crop_rectangle[0] : crop_rectangle[2]
    ]
    norm_back = norm_image(background)

    height, width, _ = norm_back.shape
    norm_back = cv2.resize(norm_back, (width * 4, height * 4))
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
        frame = cv2.resize(frame, (width * 4, height * 4))
        filtered = cv2.resize(filtered, (width * 4, height * 4))

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

        cv2.imshow("thermal", frame)
        cv2.imshow("filtered", filtered)
        cv2.moveWindow("thermal", 0, 0)
        cv2.moveWindow("filtered", width * 5, 0)
        cv2.waitKey(10)
        frame_i += 1


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
