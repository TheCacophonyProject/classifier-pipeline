import threading
import logging
import json
import numpy as np
from pydbus import SystemBus
from gi.repository import GLib
from ml_tools.tools import CustomJSONEncoder

SNAPSHOT_NAME = "still.png"
DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


class Service(object):
    """
    <node>
        <interface name='org.cacophony.thermalrecorder'>
            <method name='TakeSnapshot'>
                <arg type='i' name='last_frame' direction='in'/>
                <arg type='(aaq(xsiqddxb)s)' name='response' direction='out'/>

            </method>
            <method name='CameraInfo'>
                <arg type='a{si}' name='response' direction='out'/>
            </method>
        </interface>
    </node>
    """

    def __init__(self, get_frame, headers):
        self.get_frame = get_frame
        self.headers = headers

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

    def TakeSnapshot(self, last_num):
        last_frame, track_meta, f_num = self.get_frame()
        logging.debug("Frame requested %s latest frame %s", last_num, f_num)

        if f_num == last_num or last_frame is None:
            return (np.empty((0, 0)), (), "")
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
            json.dumps(track_meta, cls=tools.CustomJSONEncoder),
        )


class SnapshotService:
    def __init__(self, get_frame, headers):
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
            args=(get_frame, headers),
        )
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self, get_frame, headers):
        bus = SystemBus()
        service = bus.publish(DBUS_NAME, Service(get_frame, headers))
        self.loop.run()
        service.unpublish()
