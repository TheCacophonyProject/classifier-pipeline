"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import pytz
import time
from datetime import datetime

from track.cliptracker import ClipTracker
import logging


class ClipTrackExtractor(ClipTracker):
    PREVIEW = "preview"
    VERSION = 11
    TYPE = "thermal"

    @property
    def tracker_version(self):
        return self.version

    @property
    def type(self):
        return ClipTrackExtractor.TYPE

    def __init__(
        self,
        config,
        cache_to_disk=False,
        keep_frames=True,
        calc_stats=True,
        verbose=False,
        do_tracking=True,
        update_background=True,
        calculate_filtered=False,
        calculate_thumbnail_info=False,
        from_pi=False,
        max_frames=None,
        model=None,
    ):
        super().__init__(
            config,
            cache_to_disk,
            keep_frames=keep_frames,
            calc_stats=calc_stats,
            verbose=verbose,
            do_tracking=do_tracking,
            calculate_thumbnail_info=calculate_thumbnail_info,
            max_frames=max_frames,
        )

        self.from_pi = from_pi
        if from_pi:
            self.version = f"PI-{ClipTrackExtractor.VERSION}"
        else:
            self.version = ClipTrackExtractor.VERSION

        self.background_alg = None
        self.update_background = update_background
        self.calculate_filtered = calculate_filtered
        self.weighting_percent = 1
        # self.cache_to_disk = cache_to_disk
        # self.max_tracks = config.max_tracks
        # # frame_padding < 3 causes problems when we get small areas...
        # self.frame_padding = max(3, self.config.frame_padding)
        # # the dilation effectively also pads the frame so take it into consideration.
        # # self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        # self.keep_frames = keep_frames
        # self.calc_stats = calc_stats
        # self._tracking_time = None
        # if self.config.dilation_pixels > 0:
        #     size = self.config.dilation_pixels * 2 + 1
        #     self.dilate_kernel = np.ones((size, size), np.uint8)
        self.model = model
        if self.config.model and model is None and do_tracking:
            logging.info("Using yolo model for object detection")
            from ml_tools.interpreter import Yolo

            self.model = Yolo(self.config.model)
        if self.model is not None:
            self.version = f"YOLO{self.model.version()}-{self.version}"

    def init_clip(self, clip):
        from cptv_rs_python_bindings import CptvReader
        from piclassifier.motiondetector import WeightedBackground

        clip.set_frame_buffer(
            self.cache_to_disk,
            self.keep_frames,
            self.max_frames,
        )
        clip.type = self.type
        reader = CptvReader(str(clip.source_file))
        header = reader.get_header()

        clip.set_res(header.x_resolution, header.y_resolution)
        if clip.from_metadata:
            for track in clip.tracks:
                track.crop_regions()
        camera_model = None
        if header.model:
            camera_model = header.model
        clip.set_model(camera_model)

        video_start_time = datetime.fromtimestamp(header.timestamp / 1000000)
        video_start_time = video_start_time.astimezone(
            pytz.timezone("Pacific/Auckland")
        )

        clip.set_video_stats(video_start_time)
        if camera_model == "lepton3.5":
            weight_add = 1 / self.weighting_percent
        else:
            weight_add = 0.1 / self.weighting_percent

        frame = reader.next_frame()
        background = frame.pix
        if self.model:
            background = self.inpaint_yolo(clip, frame.pix)
        clip.update_background(background)
        clip._background_calculated()
        self.background_alg = WeightedBackground(
            clip.crop_rectangle.x,
            clip.crop_rectangle,
            clip.res_x,
            clip.res_y,
            weight_add,
        )
        self.background_alg.process_frame(background)

    def inpaint_yolo(self, clip, thermal):
        """
        Removes yolo detected objects from thermal by otsu thresholding each detection
        box and inpainting the hot pixels, so they are not used as background
        """
        import cv2

        detections = self.model.predict(thermal)
        mask = np.zeros(thermal.shape, dtype=np.uint8)
        res_y, res_x = thermal.shape
        for det in detections:
            x = int(round(det["x"]))
            y = int(round(det["y"]))
            w = max(1, int(round(det["width"])))
            h = max(1, int(round(det["height"])))

            if (
                abs(w - clip.crop_rectangle.width) < 4
                and abs(h - clip.crop_rectangle.height) < 4
            ):
                continue
            x_end = min(res_x, x + w)
            y_end = min(res_y, y + h)
            x = max(0, x)
            y = max(0, y)
            if x_end - x < 2 or y_end - y < 2:
                continue
            box = np.float32(thermal[y:y_end, x:x_end])
            box = cv2.normalize(box, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # objects are hotter than their surroundings
            _, box_mask = cv2.threshold(box, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask[y:y_end, x:x_end] |= box_mask
        if not np.any(mask):
            return thermal
        # grow the mask to cover the cooler edges of the object
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        inpainted = cv2.inpaint(np.float32(thermal), mask, 3, cv2.INPAINT_TELEA)
        debug = False
        if debug:
            # original | mask | inpainted, original and inpainted on the same scale
            low = float(np.amin(thermal))
            high = float(np.amax(thermal))
            scale = 255 / max(1, high - low)
            original_img = np.uint8((np.float32(thermal) - low) * scale)
            inpainted_img = np.uint8(np.clip((inpainted - low) * scale, 0, 255))
            comparison = np.hstack((original_img, mask * 255, inpainted_img))
            comparison = cv2.resize(
                comparison, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("i", comparison)
            cv2.moveWindow("i", x, 300)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return inpainted.astype(thermal.dtype)

    def parse_clip(self, clip, process_background=False):
        """
        Loads a cptv file, and prepares for track extraction.
        """

        self._tracking_time = None
        start = time.time()
        self.init_clip(clip)
        self._track_clip(clip, process_background=process_background)
        if self.calc_stats:
            clip.stats.completed()
        self._tracking_time = time.time() - start
        return True

    def _track_clip(self, clip, process_background=False):

        if clip.background is None:
            logging.error("Clip has no background have you called init_clip first")
            raise Exception("Clip has no background have you called init_clip first")
        from cptv_rs_python_bindings import CptvReader

        reader = CptvReader(str(clip.source_file))
        while True:
            frame = reader.next_frame()
            if frame is None:
                break

            if not process_background and frame.background_frame:
                continue
            self.process_frame(clip, frame)
            if self.update_background or self.background_alg.background is None:
                # use mean of last 45 frames to update background, this will help
                # when pixels become cooler for a very short time i.e. tracked object is cooler than background
                last_avg = np.mean(
                    [f.thermal for f in clip.frame_buffer.get_last_x(x=45)], axis=0
                )
                self.background_alg.process_frame(last_avg)
        if not clip.from_metadata and self.do_tracking:
            self.apply_track_filtering(clip)

    @property
    def tracking_time(self):
        return self._tracking_time

    def start_tracking(self, clip, frames, track_frames, background_alg=None, **args):
        # no need to retrack all of preview
        do_tracking = self.do_tracking
        self.do_tracking = False
        self.background_alg = background_alg
        new_tracks = []
        tracking_start = len(frames) - track_frames
        for i, frame in enumerate(frames):
            if not self.do_tracking and i >= tracking_start:
                self.do_tracking = do_tracking
            new_t, _ = self.process_frame(clip, frame)
            new_tracks.extend(new_t)
        self.do_tracking = do_tracking
        return new_tracks

    def process_frame(self, clip, frame):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        If specified background subtraction algorithm will be used.
        """
        from piclassifier.cptvmotiondetector import is_affected_by_ffc

        ffc_affected = is_affected_by_ffc(frame)
        thermal = frame.pix.copy()
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected
        mask = None
        filtered = None
        if self.do_tracking or self.calculate_filtered or self.calculate_thumbnail_info:
            filtered = np.float32(frame.pix) - self.background_alg.background
        if self.do_tracking or self.calculate_thumbnail_info:
            # done to get the mask regardless of yolo
            from ml_tools.imageprocessing import detect_objects

            obj_filtered, threshold = self._get_normalized_filtered_frame(
                clip, thermal, filtered, denoise=self.config.denoise
            )
            _, mask, component_details, centroids = detect_objects(
                obj_filtered, otsus=False, threshold=threshold, kernel=(5, 5)
            )
            if self.model:
                centroids = None
                detections = self.model.predict(thermal)
                # convert to connectedComponentsWithStats format [x, y, w, h, area]
                # plus mask_id, the most common mask label inside the box
                # yolo gives no mask so use the motion mask pixels inside the box as mass
                component_details = []
                for det in detections:
                    x = int(round(det["x"]))
                    y = int(round(det["y"]))
                    w = max(1, int(round(det["width"])))
                    h = max(1, int(round(det["height"])))
                    masked_region = mask[
                        max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)
                    ]
                    labels = masked_region[masked_region > 0]
                    mass = len(labels)
                    mask_id = int(np.bincount(labels).argmax()) if mass > 0 else None
                    component_details.append([x, y, w, h, mass, mask_id])
                    # TODO keep confidence and class info
        clip_frame = clip.add_frame(thermal, filtered, mask, ffc_affected)
        if not self.do_tracking:
            return [], []

        new_tracks = []
        stale_tracks = []
        regions = None
        if not clip.from_metadata:
            regions = []
            if ffc_affected:
                clip.active_tracks = set()
            else:
                regions = self._get_regions_of_interest(
                    clip,
                    component_details,
                    centroids=centroids,
                    source="model" if self.model else "cv2",
                )
                new_tracks, stale_tracks = self._apply_region_matchings(clip, regions)
            if not self.from_pi:
                # region_history is only consumed by offline thumbnail
                # selection (classify.thumbnail.best_trackless_thumb); on the
                # Pi it would just grow unbounded for the life of the clip
                clip.region_history.append(regions)
        debug = False
        if debug:
            debug_frame(clip_frame, regions)
        return new_tracks, stale_tracks


def debug_frame(frame, regions):
    if frame.filtered is None or frame.thermal is None:
        return
    from ml_tools.imageprocessing import normalize

    thermal, _ = normalize(frame.thermal, new_max=255)
    filtered, _ = normalize(frame.filtered, new_max=255)
    import cv2

    thermal = cv2.cvtColor(np.uint8(thermal), cv2.COLOR_GRAY2BGR)
    filtered = cv2.cvtColor(np.uint8(filtered), cv2.COLOR_GRAY2BGR)
    for region in regions:
        start = (int(region.left), int(region.top))
        end = (int(region.right), int(region.bottom))
        cv2.rectangle(thermal, start, end, (0, 0, 255), 1)
        cv2.rectangle(filtered, start, end, (0, 0, 255), 1)

    cv2.imshow("t", thermal)

    cv2.imshow("f", filtered)
    cv2.moveWindow("f", 300, 300)
    cv2.waitKey()
