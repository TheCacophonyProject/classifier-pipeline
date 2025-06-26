import threading
import logging
import json
import numpy as np
import time
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
from ml_tools.tools import CustomJSONEncoder
from cptv import Frame

from dbus.mainloop.glib import DBusGMainLoop

DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


class Service(dbus.service.Object):
    def __init__(
        self, dbus, get_frame, headers, take_snapshot_fn, labels, get_thumbnail
    ):
        super().__init__(dbus, DBUS_PATH)
        self.get_frame = get_frame
        self.get_thumbnail = get_thumbnail
        self.headers = headers
        self.take_snapshot = take_snapshot_fn
        self.labels = labels

    @dbus.service.method(
        DBUS_NAME,
        in_signature="",
        out_signature="a{si}",
    )
    def CameraInfo(self):
        logging.debug("Serving headers %s", self.headers)
        headers = self.headers.as_dict()
        ir = headers.get("model") == "IR"
        for k, v in headers.items():
            try:
                headers[k] = int(v)
            except:
                headers[k] = 0
                pass
        headers["FPS"] = headers.get("fps", 9)
        headers["ResX"] = headers.get("res_x", 160)
        headers["ResY"] = headers.get("res_y", 120)
        if ir:
            headers["Model"] = 2
        else:
            headers["Model"] = 1
        logging.debug("Sending headers %s", headers)
        return headers

    @dbus.service.method(
        DBUS_NAME,
        in_signature="i",
        out_signature="(aaq(xsiqddxb)s)",
    )
    def TakeSnapshot(self, last_num):
        s = time.time()
        last_frame, track_meta, f_num = self.get_frame(last_num)

        if f_num == last_num or last_frame is None:
            return (np.empty((0, 0)), (0, "", f_num, 0, 0, 0, 0, False), "")
        logging.debug(
            "Frame requested %s latest frame %s took %s",
            last_num,
            f_num,
            time.time() - s,
        )
        if not isinstance(last_frame, Frame):
            last_frame = last_frame[:, :, 0]
            return (
                last_frame,
                (0, "", f_num, 0, 0, 0, 0, 0),  # count
                json.dumps(track_meta, cls=CustomJSONEncoder),
            )
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

    @dbus.service.method(
        DBUS_NAME,
        out_signature="aaqiai",
    )
    def GetThumbnail(self, clip_id, track_id):
        if track_id == 0:
            track_id = None
        if clip_id == 0:
            clip_id = None
        result = self.get_thumbnail(clip_id, track_id)
        if result is None:
            raise Exception("No thumbnail")
        thumb, track_id, region = result

        return thumb, track_id, region.to_ltrb()

    @dbus.service.method(
        DBUS_NAME,
    )
    def TakeTestRecording(self):
        logging.info("Take test recording")
        result = False
        try:
            result = self.take_snapshot()
        except:
            logging.error("Error taking test recording", exc_info=True)

        return result

    @dbus.service.method(DBUS_NAME, signature="a{ias}")
    def ClassificationLabels(self):
        logging.info("Getting labels %s", self.labels)
        return self.labels

    @dbus.service.signal(DBUS_NAME, signature="iiaisiaiiibbii")
    def Tracking(
        self,
        clip_id,
        track_id,
        prediction,
        what,
        confidence,
        region,
        frame,
        mass,
        blank,
        tracking,
        last_prediction_frame,
        model_id,
    ):
        pass

    @dbus.service.signal(DBUS_NAME, signature="xb")
    def Recording(self, timestamp, is_recording):
        pass


class SnapshotService:
    def __init__(self, get_frame, headers, take_snapshot_fn, labels, get_thumbnail):
        DBusGMainLoop(set_as_default=True)
        dbus.mainloop.glib.threads_init()
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
            args=(get_frame, headers, take_snapshot_fn, labels, get_thumbnail),
        )
        self.t.start()
        self.service = None

    def quit(self):
        self.loop.quit()

    def run_server(self, get_frame, headers, take_snapshot_fn, labels, get_thumbnail):
        session_bus = dbus.SystemBus(mainloop=DBusGMainLoop())
        name = dbus.service.BusName(DBUS_NAME, session_bus)
        self.service = Service(
            session_bus, get_frame, headers, take_snapshot_fn, labels, get_thumbnail
        )
        self.loop.run()

    def tracking(
        self,
        clip_id,
        track,
        prediction,
        region,
        tracking,
        last_prediction_frame,
        labels,
        model_id,
    ):
        logging.debug(
            "Tracking?  %s region %s prediction %s track %s",
            tracking,
            region,
            prediction,
            track.get_id(),
        )
        if self.service is None:
            return
        if prediction is not None:
            predictions = prediction.copy()
            predictions = np.uint8(np.round(predictions * 100))
            best = np.argmax(predictions)
            self.service.Tracking(
                clip_id,
                track.get_id(),
                predictions,
                labels[best],
                predictions[best],
                region.to_ltrb(),
                region.frame_number,
                region.mass,
                region.blank,
                tracking,
                last_prediction_frame,
                model_id,
            )
        else:
            self.service.Tracking(
                clip_id,
                track.get_id(),
                [],
                "",
                0,
                region.to_ltrb(),
                region.frame_number,
                region.mass,
                region.blank,
                tracking,
                last_prediction_frame,
                0,
            )

    def recording(self, is_recording):
        if self.service is None:
            return
        self.service.Recording(np.int64(time.time()), is_recording)
