import dbus
import logging
import binascii
from dbus.mainloop.glib import DBusGMainLoop

DBUS_NAME = "org.cacophony.i2c"
DBUS_PATH = "/org/cacophony/i2c"


def set_recording_state(is_recording):
    bus = dbus.SystemBus(mainloop=DBusGMainLoop())
    try:
        proxy = bus.get_object(DBUS_NAME, DBUS_PATH)
        state = agent_state(proxy)
        if state is None:
            return False
        state = int(state[0])
        if is_recording:
            state = state | 4
        else:
            state = state & ~4

        res = agent_state(proxy, state)
        if res is None:
            return False

    except:
        logging.error("set recording dbus error ", exc_info=True)
        return False
    return True


def agent_state(proxy, value=None):
    if value is None:
        payload = bytearray([7])
    else:
        payload = bytearray([7, value])

    crc = binascii.crc_hqx(payload, 0x1D0F)
    crc = crc.to_bytes(2, "big")
    payload.extend(crc)

    try:
        return proxy.Tx(0x25, dbus.ByteArray(payload), 3 if value is None else 0, 1000)
    except:
        logging.error("agent state dbus error ", exc_info=True)
    return None
