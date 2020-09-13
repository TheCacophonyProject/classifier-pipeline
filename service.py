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

    def __init__(self, processor):
        self.processor = processor

    def TakeSnapshot(self):
        last_frame = self.processor.get_recent_frame()
        if last_frame is None:
            return "Reading from camera has not start yet."

        frame_to_jpg(last_frame, self.processor.output_dir + "/" + SNPASHOT_NAME)
        return "Success"


class SnapshotService:
    def __init__(self, processor):
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(target=self.run_server, args=(processor,))
        self.t.start()

    def quit(self):
        self.loop.quit()

    def run_server(self, processor):
        bus = SystemBus()
        service = bus.publish(DBUS_NAME, Service(processor))
        self.loop.run()
        service.unpublish()
