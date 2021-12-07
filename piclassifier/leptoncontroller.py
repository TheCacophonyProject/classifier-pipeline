import dbus
import logging

DBUS_NAME = "org.cacophony.leptond"
DBUS_PATH = "/org/cacophony/leptond"


def get_dbus_obj():
    bus = dbus.SystemBus()
    try:
        dbus_o = bus.get_object(DBUS_NAME, DBUS_PATH)
        return dbus_o
    except:
        logging.error("Error getting lepton dbus", exc_info=True)
    return None


def set_auto_ffc(automatic):
    obj = get_dbus_obj()
    if obj is None:
        return False
    return obj.SetAutoFFC(automatic)


def run_ffc():
    obj = get_dbus_obj()
    if obj is None:
        return false
    return obj.RunFFC()
