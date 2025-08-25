#!/usr/bin/env python3

import sys
from gi.repository import GLib

import dbus
import dbus.mainloop.glib
import time
import threading
import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from ml_tools.logs import init_logging


DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


model_labels = []
active_tracks = {}


def tracking(
    clip_id,
    track_id,
    prediction,
    what,
    confidence,
    region,
    frame,
    mass,
    blank,
    tracking,
    last_prediction_frame,
    model_id,
):
    logging.info(
        "Received tracking event for clip %s track %s prediction of %s with %s%% confidence still tracking ? %s at region %s",
        clip_id,
        track_id,
        what,
        confidence,
        tracking,
        region,
    )


def normalize(thumb):
    a_max = np.amax(thumb)
    a_min = np.amin(thumb)
    return np.uint8(255 * (np.float32(thumb) - a_min) / (a_max - a_min))


def recording(dt, is_recording):
    if is_recording:
        logging.info("Recording started at %s", datetime.fromtimestamp(dt))
    else:
        logging.info("Recording ended at %s", datetime.fromtimestamp(dt))


def track_filtered(clip_id, track_id):
    logging.info("Clip %s has filtered track %s", clip_id, track_id)


def tracking_reprocessed(
    clip_id,
    track_id,
    prediction,
    what,
    confidence,
    region,
    frame,
    mass,
    blank,
    tracking,
    last_prediction_frame,
    model_id,
    rec_end,
):
    rec_end = datetime.fromtimestamp(rec_end)
    logging.info(
        "Received post processing event for recording ended at %s clip %s track %s prediction of %s with %s%% confidence at region %s",
        rec_end,
        clip_id,
        track_id,
        what,
        confidence,
        region,
    )

    bus = dbus.SystemBus()
    dbus_object = bus.get_object(DBUS_NAME, DBUS_PATH)
    try:
        thumb, track_id, region = dbus_object.GetThumbnail(clip_id, track_id)
        thumb = np.uint16(thumb)
        thumb = normalize(thumb)
        cv2.imwrite(f"./thumbnails/{rec_end}-{clip_id}-{track_id}.png", thumb)
        logging.info(
            "Saved thumb to %s ", f"./thumbnails/{rec_end}-{clip_id}-{track_id}.png"
        )
    except:
        logging.info(
            "Could not get thumbnail for %s %s", clip_id, track_id, exc_info=True
        )


# helper class to run dbus in background
class TrackingService:
    def __init__(self):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
        )
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self):
        dbus_object = None
        try:
            bus = dbus.SystemBus()
            dbus_object = bus.get_object(DBUS_NAME, DBUS_PATH)
        except dbus.exceptions.DBusException as e:
            print("Failed to initialize D-Bus object: '%s'" % str(e))
            sys.exit(2)
        global model_labels
        model_labels = dbus_object.ClassificationLabels()

        bus.add_signal_receiver(
            tracking,
            dbus_interface=DBUS_NAME,
            signal_name="Tracking",
        )
        bus.add_signal_receiver(
            recording,
            dbus_interface=DBUS_NAME,
            signal_name="Recording",
        )

        bus.add_signal_receiver(
            tracking_reprocessed,
            dbus_interface=DBUS_NAME,
            signal_name="TrackingReprocessed",
        )
        bus.add_signal_receiver(
            track_filtered,
            dbus_interface=DBUS_NAME,
            signal_name="TrackFiltered",
        )
        self.loop.run()


def main():
    init_logging()
    thumb_dir = Path("./thumbnails")
    thumb_dir.mkdir(exist_ok=True)
    service = TrackingService()

    # just to keep program alive
    while service.t.is_alive():
        time.sleep(10)


if __name__ == "__main__":
    main()
