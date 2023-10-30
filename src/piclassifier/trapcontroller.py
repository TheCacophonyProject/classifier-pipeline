import time
import dbus
import logging
import json

DBUS_NAME = "org.cacophony.TrapController"
DBUS_PATH = "/org/cacophony/TrapController"


def trigger_trap(tag=None):
    bus = dbus.SystemBus()
    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        data = {}
        if tag is not None:
            data["tag"] = tag
        proxy.TriggerTrap(json.dumps(data))
    except:
        logging.error("trapped dbus error ", exc_info=True)
