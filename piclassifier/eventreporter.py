import time
from pydbus import SystemBus

DBUS_NAME = "org.cacophony.Events"
DBUS_PATH = "/org/cacophony/Events"

def throttled_event():
    bus = SystemBus()
    proxy = bus.get(DBUS_NAME,DBUS_PATH)

    proxy.Add("{}","throttle",time.time_ns())
