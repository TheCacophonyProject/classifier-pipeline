import threading
import os

from ml_tools.tools import frame_to_jpg

from pydbus import SystemBus
from gi.repository import GLib

SNPASHOT_NAME = "still.png"
DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


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
        last_frame = self.get_frame()
        if last_frame is None:
            return "Reading from camera has not start yet."

        frame_to_jpg(last_frame, self.output_dir + "/" + SNPASHOT_NAME)
        return "Success"


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
