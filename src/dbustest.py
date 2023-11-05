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


def catchall_tracking_signals_handler(what, confidence, region, tracking):
    print(
        "Received a trackng signal and it says " + what,
        confidence,
        "% at ",
        region,
        " tracking?",
        tracking,
    )


# helper class to run dbus in background
class TrackingService:
    def __init__(self, callback):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.callback = callback
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
        )
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self):
        try:
            bus = dbus.SystemBus()
            object = bus.get_object(DBUS_NAME, DBUS_PATH)
        except dbus.exceptions.DBusException as e:
            print("Failed to initialize D-Bus object: '%s'" % str(e))
            sys.exit(2)

        bus.add_signal_receiver(
            self.callback,
            dbus_interface=DBUS_NAME,
            signal_name="Tracking",
        )
        self.loop.run()


if __name__ == "__main__":
    tracking = TrackingService(catchall_tracking_signals_handler)

    # just to keep program alive
    # replace with your code
    while tracking.t.is_alive():
        time.sleep(1)
