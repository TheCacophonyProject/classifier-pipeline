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


def log_event(event_type, details):
    bus = dbus.SystemBus()
    data = {"description": {"details": details}}
    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        json_d = json.dumps(data)
        proxy.Add(json_d, event_type, dbus.Int64(time.time_ns()))
    except:
        logging.error("log event dbus error ", exc_info=True)
