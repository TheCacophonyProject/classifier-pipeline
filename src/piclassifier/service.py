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
from dbus.exceptions import DBusException
from dbus.mainloop.glib import DBusGMainLoop

DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


class ParseFileError(dbus.exceptions.DBusException):
    _dbus_error_name = DBUS_NAME + ".ParseFileError"


class Service(dbus.service.Object):
    def __init__(
        self,
        get_frame,
        headers,
        take_snapshot_fn,
        labels,
        get_thumbnail,
        thumbnail_dir,
        parse_file,
        is_parsing_file,
        classifier_loaded=True,
    ):
        self.get_frame = get_frame
        self.get_thumbnail = get_thumbnail
        self.headers = headers
        self.take_snapshot = take_snapshot_fn
        self.labels = labels
        self.thumbnail_dir = thumbnail_dir
        self.parse_file = parse_file
        self.is_parsing_file = is_parsing_file
        self.classifier_loaded = classifier_loaded

    def start_service(self,dbus):
        super().__init__(dbus, DBUS_PATH)
        self.ServiceStarted()


    def update_labels(self, labels):
        self.labels = labels
        self.classifier_loaded = True
        try:
            self.LabelsUpdated()
        except:
            logging.error("Could run labels updated",exc_info=True)
        
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
        out_signature="s",
    )
    def ParsingFile(self):
        parsing_file = self.is_parsing_file()
        return parsing_file if parsing_file is not None else ""

    @dbus.service.method(
        DBUS_NAME,
        in_signature="sii",
    )
    def ParseFile(self, file, fps, seed):
        parsing_file = self.is_parsing_file()
        if parsing_file is not None:
            raise ParseFileError(f"Already parsing {parsing_file}")
        threading.Thread(
            target=self.parse_file, args=(file, fps, seed), daemon=True
        ).start()
        return "Parsing file"

    @dbus.service.method(
        DBUS_NAME,
        in_signature="i",
        out_signature="(aaq(xsiqddxb)s)",
    )
    def TakeSnapshot(self, last_num):

        from cptv import Frame

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
            # check thumbnail dir
            thumb_file = self.thumbnail_dir / f"{clip_id}-{track_id}.npy"
            if thumb_file.exists():
                thumb = np.load(thumb_file)
                # dont think any need for region here
                region = []
            else:
                raise Exception("No thumbnail")
        else:
            thumb = result.thumb
            track_id = result.track_id
            region = result.region
            region = region.to_ltrb()

        return thumb, track_id, region

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
        if not self.classifier_loaded:
            raise DBusException("Labels have not been initialized")
        logging.info("Getting labels %s", self.labels)
        if len(self.labels) == 0:
            return dbus.Array([], signature="(ias)")
        return self.labels

    @dbus.service.signal(DBUS_NAME, signature="iiaisiaiiibbisx")
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
        track_start_time,
    ):
        pass

    @dbus.service.signal(DBUS_NAME, signature="ii")
    def TrackFiltered(self, clip_id, track_id):
        pass

    @dbus.service.signal(DBUS_NAME, signature="xb")
    def Recording(self, timestamp, is_recording):
        pass

    @dbus.service.method(DBUS_NAME, in_signature="iiaisiaiiibbisx")
    def TrackReprocessed(
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
        clip_end_time,
    ):
        # just passing on the tracking info
        return self.TrackingReprocessed(
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
            str(model_id),
            dbus.Int64(clip_end_time),
        )
        # pass

    @dbus.service.signal(DBUS_NAME, signature="iiaisiaiiibbisx")
    def TrackingReprocessed(
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
        clip_end_time,
    ):
        pass

    @dbus.service.signal(DBUS_NAME)
    def LabelsUpdated(self):
        pass

    @dbus.service.signal(DBUS_NAME)
    def ServiceStarted(self):
        pass


class SnapshotService:
    def __init__(
        self,
        get_frame,
        headers,
        take_snapshot_fn,
        labels,
        get_thumbnail,
        thumbnail_dir,
        parse_file,
        is_parsing_file,
        classifier_loaded=True,
    ):
        DBusGMainLoop(set_as_default=True)
        dbus.mainloop.glib.threads_init()
        self.loop = GLib.MainLoop()
       
        self.service = Service(
            get_frame,
            headers,
            take_snapshot_fn,
            labels,
            get_thumbnail,
            thumbnail_dir,
            parse_file,
            is_parsing_file,
            classifier_loaded,
        )
    
        self.t = threading.Thread(
            target=self.run_server,
        )
        self.t.daemon = True
        self.t.start()
    

    def update_service(
        self,
        get_frame,
        headers,
        take_snapshot_fn,
        labels,
        get_thumbnail,
        thumbnail_dir,
        parse_file,
    ):
        self.service.get_frame = get_frame
        self.service.headers = headers
        self.service.take_snapshot = take_snapshot_fn
        self.service.labels = labels
        self.service.get_thumbnail = get_thumbnail
        self.service.thumbnail_dir = thumbnail_dir
        self.service.parse_file = parse_file

    def quit(self):
        self.loop.quit()
        self.service = None

    def run_server(
        self,
    ):
        try:
            session_bus = dbus.SystemBus(mainloop=DBusGMainLoop())
            name = dbus.service.BusName(DBUS_NAME, session_bus)
            self.service.start_service(session_bus)
            self.loop.run()
        except:
            logging.error("Couldn't run loop",exc_info=True)

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
        track_start_time,
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
                str(model_id),
                int(track_start_time * 1000),  # convert to ms
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
                "0",
                int(track_start_time * 1000),  # convert to ms
            )

    def track_filtered(self, clip_id, track_id):
        if self.service is None:
            return
        self.service.TrackFiltered(clip_id, track_id)

    def recording(self, epoch_time, is_recording):
        if self.service is None:
            return
        # convert to ms
        self.service.Recording(dbus.Int64(int(epoch_time * 1000)), is_recording)
