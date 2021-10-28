import threading
import logging
import json
import numpy as np

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
from ml_tools.tools import CustomJSONEncoder

DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


class Service(dbus.service.Object):
    def __init__(self, dbus, get_frame, headers):
        super().__init__(dbus, DBUS_PATH)
        self.get_frame = get_frame
        self.headers = headers

    @dbus.service.method(
        "org.cacophony.thermalrecorder",
        in_signature="",
        out_signature="a{si}",
    )
    def CameraInfo(self):
        logging.debug("Serving headers %s", self.headers)
        headers = self.headers.as_dict()
        for k, v in headers.items():
            try:
                headers[k] = int(v)
            except:
                headers[k] = 0
                pass
        headers["FPS"] = headers.get("fps", 9)
        headers["ResX"] = headers.get("res_x", 160)
        headers["ResY"] = headers.get("res_y", 120)

        logging.debug("Sending headers %s", headers)
        return headers

    @dbus.service.method(
        "org.cacophony.thermalrecorder",
        in_signature="i",
        out_signature="(aaq(xsiqddxb)s)",
    )
    def TakeSnapshot(self, last_num):
        last_frame, track_meta, f_num = self.get_frame()
        logging.debug("Frame requested %s latest frame %s", last_num, f_num)

        if f_num == last_num or last_frame is None:
            return (np.empty((0, 0)), (0, "", 0, 0, 0, 0, 0, False), "")
        return (
            last_frame.pix,
            (
                last_frame.time_on.total_seconds() * 1e9,
                "",
                f_num,  # count
                0,
                last_frame.temp_c,
                last_frame.last_ffc_temp_c,
                last_frame.last_ffc_time.total_seconds() * 1e9,
                last_frame.background_frame,
            ),
            json.dumps(track_meta, cls=CustomJSONEncoder),
        )


class SnapshotService:
    def __init__(self, get_frame, headers):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
            args=(get_frame, headers),
        )
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self, get_frame, headers):
        session_bus = dbus.SystemBus()
        name = dbus.service.BusName(DBUS_NAME, session_bus)
        object = Service(session_bus, get_frame, headers)
        self.loop.run()
