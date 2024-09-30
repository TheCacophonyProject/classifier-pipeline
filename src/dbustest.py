#!/usr/bin/env python3

# Take in a single optional integral argument
import sys

DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"
# Create a reference to the RandomData object on the  session bus
from gi.repository import GLib

import dbus
import dbus.mainloop.glib
import time
import threading
from datetime import datetime

labels = []


def catchall_tracking_signals_handler(
    prediction,
    what,
    confidence,
    region,
    frame,
    mass,
    blank,
    tracking,
    last_prediction_frame,
):
    print(
        "Received a trackng signal and it says " + what,
        confidence,
        "% at ",
        region,
        " tracking?",
        tracking,
        "prediction",
        prediction,
        "frame",
        frame,
        "mass",
        mass,
        "blank",
        blank,
        "last prediction frame",
        last_prediction_frame,
    )
    index = 0
    for x in prediction:
        print("For  ", labels[index], " have confidence ", int(x), "%")
        index += 1


def catchall_rec_signals_handler(dt, is_recording):
    if is_recording:
        print("Recording started at ", datetime.fromtimestamp(dt))
    else:
        print("Recording ended at ", datetime.fromtimestamp(dt))


# helper class to run dbus in background
class TrackingService:
    def __init__(self, callback, rec_callback):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.callback = callback
        self.rec_callback = rec_callback
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
        global labels
        labels = dbus_object.ClassificationLabels()
        print("Labels are ", labels)
        bus.add_signal_receiver(
            self.callback,
            dbus_interface=DBUS_NAME,
            signal_name="Tracking",
        )
        bus.add_signal_receiver(
            self.rec_callback,
            dbus_interface=DBUS_NAME,
            signal_name="Recording",
        )
        self.loop.run()


if __name__ == "__main__":
    tracking = TrackingService(
        catchall_tracking_signals_handler, catchall_rec_signals_handler
    )

    # just to keep program alive
    # replace with your code
    while tracking.t.is_alive():
        time.sleep(1)
