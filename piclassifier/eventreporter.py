import time
import dbus
import logging
import json

DBUS_NAME = "org.cacophony.Events"
DBUS_PATH = "/org/cacophony/Events"

def throttled_event():
    bus = dbus.SystemBus()
    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        proxy.Add("{}", "throttle", dbus.Int64(time.time_ns()))
    except:
        logging.error("throttle dbus error ", exc_info=True)
