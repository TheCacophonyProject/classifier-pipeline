import threading
import logging
from pydbus import SystemBus
from gi.repository import GLib

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
                <arg type='a{ss}' name='response' direction='out'/>
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
            headers[k] = "{}".format(v)
        logging.debug("Sending headers %s", headers)
        return headers

    def TakeSnapshot(self, last_frame):
        last_frame, track_meta, f_num = self.get_frame()
        logging.debug("Frame requested %s latest frame %s", last_frame, f_num)

        if f_num == last_frame:
            return None
        return (
            last_frame.pix,
            (
                last_frame.time_on,
                "",
                f_num,  # count
                0,
                last_frame.temp_c,
                last_frame.last_ffc_temp_c,
                last_frame.last_ffc_time,
                last_frame.background_frame,
            ),
            json.dumps(f_num),
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
