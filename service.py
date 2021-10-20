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
        return self.headers

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

    def run_server(self, get_frame):
        bus = SystemBus()
        service = bus.publish(DBUS_NAME, Service(get_frame))
        self.loop.run()
        service.unpublish()


class Service(object):
    """
    <node>
        <interface name='org.cacophony.thermalrecorder'>
            <method name='TakeSnapshot'>
                <arg type='s' name='response' direction='out'/>
            </method>
        </interface>
    </node>
    """

    def __init__(self, get_frame, output_dir):
        self.get_frame = get_frame
        self.output_dir = output_dir

    def TakeSnapshot(self):
        last_frame, track_meta = self.get_frame()
        if last_frame is None:
            return "Reading from camera has not start yet."
        #
        # last_frame.save(self.output_dir + "/" + SNAPSHOT_NAME)
        # # frame_to_jpg(last_frame, self.output_dir + "/" + SNPASHOT_NAME)
        return {
            "Pix": last_frame.pix,
            "Status": {"FrameCount": last_frame.frame_counter},
        }
        # return "Success"


class SnapshotService:
    def __init__(self, get_frame, output_dir):
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
            args=(
                get_frame,
                output_dir,
            ),
        )
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self, get_frame, output_dir):
        bus = SystemBus()
        service = bus.publish(DBUS_NAME, Service(get_frame, output_dir))
        self.loop.run()
        service.unpublish()
