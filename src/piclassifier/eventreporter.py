import time
import dbus
import logging
import json
from dbus.mainloop.glib import DBusGMainLoop

DBUS_NAME = "org.cacophony.Events"
DBUS_PATH = "/org/cacophony/Events"


def throttled_event():
    bus = dbus.SystemBus(mainloop=DBusGMainLoop())
    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        proxy.Add("{}", "throttle", dbus.Int64(time.time_ns()))
    except:
        logging.error("throttle dbus error ", exc_info=True)


def log_event(event_type, details=None):
    bus = dbus.SystemBus(mainloop=DBusGMainLoop())
    data = None
    if details is not None:
        data = {"description": {"details": details}}

    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        if data is None:
            json_d = "{}"
        else:
            json_d = json.dumps(data,cls=CustomJSONEncoder)
        proxy.Add(json_d, event_type, dbus.Int64(time.time_ns()))
    except:
        logging.error("log event dbus error ", exc_info=True)


class CustomJSONEncoder(json.JSONEncoder):
    import numpy as np
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
