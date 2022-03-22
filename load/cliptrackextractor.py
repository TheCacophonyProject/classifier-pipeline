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
import matplotlib.pyplot as plt

import os
import logging
import numpy as np
import time
import yaml
from datetime import datetime

from cptv import CPTVReader
import cv2

from .clip import Clip
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track
from piclassifier.motiondetector import is_affected_by_ffc
<<<<<<< HEAD
from ml_tools.imageprocessing import (
    detect_objects,
    normalize,
    detect_objects_ir,
    theshold_saliency,
    detect_objects_both,
)


class ClipTrackExtractor:
    # BASE_DISTANCE_CHANGE = 450
    # more res more distnace
    BASE_DISTANCE_CHANGE = 11250

    # minimum region mass change
    MIN_MASS_CHANGE = 20 * 4
    # enforce mass growth after X seconds
    RESTRICT_MASS_AFTER = 1.5
    # amount region mass can change
    MASS_CHANGE_PERCENT = 0.55

    # MAX_DISTANCE = 2000
    MAX_DISTANCE = 30752
=======
from ml_tools.imageprocessing import detect_objects, normalize
from track.cliptracker import ClipTracker


class ClipTrackExtractor(ClipTracker):

>>>>>>> gp-master
    PREVIEW = "preview"
    VERSION = 10

    @property
    def tracker_version(self):
        return f"ClipTrackExtractor-{ClipTrackExtractor.VERSION}"

    def __init__(
        self,
        config,
        use_opt_flow,
        cache_to_disk,
        keep_frames=True,
        calc_stats=True,
        high_quality_optical_flow=False,
        verbose=False,
    ):
<<<<<<< HEAD
        self.saliency = None
        self.verbose = verbose
        self.config = config
=======
        super().__init__(
            config,
            cache_to_disk,
            keep_frames=keep_frames,
            calc_stats=calc_stats,
            verbose=verbose,
        )
>>>>>>> gp-master
        self.use_opt_flow = use_opt_flow
        self.high_quality_optical_flow = high_quality_optical_flow
        self.cache_to_disk = cache_to_disk
        self.max_tracks = config.max_tracks
        # frame_padding < 3 causes problems when we get small areas...
        self.frame_padding = max(3, self.config.frame_padding)
        # the dilation effectively also pads the frame so take it into consideration.
        self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        self.keep_frames = keep_frames
        self.calc_stats = calc_stats
        self._tracking_time = None
        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            self.dilate_kernel = np.ones((size, size), np.uint8)

    def parse_clip(self, clip, process_background=False):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.tracking_time = None
        start = time.time()
        clip.set_frame_buffer(
            self.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )
        _, ext = os.path.splitext(clip.source_file)
        count = 0
        movement_mask = None
        background = None
        if ext != ".cptv":
            vidcap = cv2.VideoCapture(clip.source_file)
            frames = []
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frames.append(gray)

                if count == 0:
                    background = gray
                    self.saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                    self.saliency.setImagesize(gray.shape[1], gray.shape[0])
                    self.saliency.init()
                    clip.set_res(gray.shape[1], gray.shape[0])
                    clip.set_model("ir")
                    clip.set_video_stats(datetime.now())

                else:
                    background = np.minimum(background, gray)
                repeats = 1
                if count < 6:
                    repeats = 6

                # filtered, _ = self._get_filtered_frame_ir(clip, gray, repeats)
                #
                # if np.amin(filtered) != 255:
                #     # filtered = cv2.dilate(filtered, (15, 15), iterations=3)
                #     if movement_mask is None:
                #         movement_mask = filtered
                #     else:
                #         movement_mask = movement_mask | filtered

                # # print("inapinting")
                # cv2.imshow("back", movement_mask)
                # # cv2.imshow("filtered", filtered)
                # #
                # cv2.waitKey(30)
                clip.current_frame += 1

                count += 1
            clip.current_frame = -1

            vidcap.release()
            # background[:] = cv2.inpaint(
            #     np.float32(background), np.uint8(movement_mask), 10, cv2.INPAINT_TELEA
            # )
            if background is None:
                return False
            background = cv2.GaussianBlur(background, (15, 15), 0)
            # cv2.imshow("backg", background)
            # cv2.waitKey(100)
            clip.update_background(background)
            for gray in frames:
                self.process_frame(clip, gray)
            vidcap.release()
        else:
            with open(clip.source_file, "rb") as f:
                reader = CPTVReader(f)
                clip.set_res(reader.x_resolution, reader.y_resolution)
                if clip.from_metadata:
                    for track in clip.tracks:
                        track.crop_regions()
                camera_model = None
                if reader.model:
                    camera_model = reader.model.decode()
                clip.set_model(camera_model)

                # if we have the triggered motion threshold should use that
                # maybe even override dynamic threshold with this value
                if reader.motion_config:
                    motion = yaml.safe_load(reader.motion_config)
                    temp_thresh = motion.get("triggeredthresh")
                    if temp_thresh:
                        clip.temp_thresh = temp_thresh

                video_start_time = reader.timestamp.astimezone(Clip.local_tz)
                clip.set_video_stats(video_start_time)
                clip.calculate_background(reader)

            with open(clip.source_file, "rb") as f:
                reader = CPTVReader(f)
                for frame in reader:
                    if not process_background and frame.background_frame:
                        continue
                    self.process_frame(clip, frame.pix, is_affected_by_ffc(frame))

        if not clip.from_metadata:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed(clip.current_frame, clip.res_y, clip.res_x)
        self.tracking_time = time.time() - start
        return True

<<<<<<< HEAD
    def process_frame(self, clip, frame, ffc_affected=False):
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected

        self._process_frame(clip, frame, ffc_affected)

    def apply_track_filtering(self, clip):
        self.filter_tracks(clip)
        # apply smoothing if required
        if self.config.track_smoothing and clip.current_frame > 0:
            for track in clip.active_tracks:
                track.smooth(Rectangle(0, 0, clip.res_x, clip.res_y))

    def _get_filtered_frame_ir(self, clip, thermal, repeats=1):
        for _ in range(repeats):
            (success, saliencyMap) = self.saliency.computeSaliency(thermal)
        # (success, saliencyMap) = self.saliency.computeSaliency(thermal)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        # cv2.imshow("saliencyMap.png", np.uint8(saliencyMap))
        return saliencyMap, 0

    # merge all regions that the midpoint is within the max(width,height) from the midpoint of another region
    # keep merging until no more merges are possible, tihs works paticularly well from the IR videos where
    # the filtered image is quite fragmented
    def merge_components(self, rectangles):
        MAX_GAP = 40
        rect_i = 0
        rectangles = list(rectangles)
        while rect_i < len(rectangles):
            rect = rectangles[rect_i]
            merged = False
            mid_x = rect[2] / 2.0 + rect[0]
            mid_y = rect[3] / 2.0 + rect[1]
            index = 0
            while index < len(rectangles):
                r_2 = rectangles[index]
                if r_2[0] == rect[0]:
                    index += 1
                    continue
                r_mid_x = r_2[2] / 2.0 + r_2[0]
                r_mid_y = r_2[3] / 2.0 + r_2[1]
                distance = (mid_x - r_mid_x) ** 2 + (r_mid_y - mid_y) ** 2
                distance = distance ** 0.5

                # widest = max(rect[2], rect[3])
                # hack short cut just take line from mid points as shortest distance subtract biggest width or hieght from each
                distance = (
                    distance - max(rect[2], rect[3]) / 2.0 - max(r_2[2], r_2[3]) / 2.0
                )
                within = r_2[0] > rect[0] and (r_2[0] + r_2[2]) <= (rect[0] + rect[2])
                within = (
                    within
                    and r_2[1] > rect[1]
                    and (r_2[1] + r_2[3]) <= (rect[1] + rect[3])
                )

                if distance < MAX_GAP or within:
                    rect[0] = min(rect[0], r_2[0])
                    rect[1] = min(rect[1], r_2[1])
                    rect[2] = max(rect[0] + rect[2], r_2[0] + r_2[2])
                    rect[3] = max(rect[1] + rect[3], r_2[1] + r_2[3])
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    # print("second merged ", rect)
                    merged = True
                    # break
                    del rectangles[index]
                else:
                    index += 1
                    # print("not mered", rect, r_2, distance)
            if merged:
                rect_i = 0
            else:
                rect_i += 1
        return rectangles

    def _process_frame(self, clip, thermal, ffc_affected=False):
        wait = 1
=======
    def tracking_time(self):
        return self._tracking_time

    def process_frame(self, clip, thermal, ffc_affected=False):
>>>>>>> gp-master
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        If specified background subtraction algorithm will be used.
        """
<<<<<<< HEAD
        repeats = 1
        if clip.current_frame < 6:
            repeats = 8
        saliencyMap, _ = self._get_filtered_frame_ir(clip, thermal, repeats=repeats)
        backsub, _ = get_filtered_frame(clip.background, thermal)
        # cv2.imshow("filtered", backsub)
        # cv2.waitKey(100)
        threshold = 0
        if np.amin(saliencyMap) == 255:
            num = 0
            mask = saliencyMap.copy()
            component_details = []
            saliencyMap[:] = 0
        else:
            num, mask, component_details = theshold_saliency(saliencyMap)
            component_details = self.merge_components(component_details[1:])
            # if clip.current_frame > 760:
            #     cv2.imshow("salfiltered.png", np.uint8(saliencyMap))
            #     cv2.imshow("backsub.png", np.uint8(backsub))
            #     print(component_details, clip.current_frame)
            #     cv2.waitKey(3000)
        # else:
        #
        #     num, mask, component_details = theshold_saliency(
        #         filtered.copy(), threshold=100
        #     )
        # backsub, _ = self._get_filtered_frame(clip, thermal)
        # num, mask, component_details = detect_objects(backsub)
        # if saliencyMap is not None:
        #     cv2.imshow("saliency", saliencyMap)
        #     cv2.imshow("backsub", np.uint8(backsub))
        #     cv2.waitKey(30)
=======
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected
        filtered, threshold = self._get_filtered_frame(clip, thermal)
        _, mask, component_details = detect_objects(
            filtered.copy(), otsus=False, threshold=threshold
        )
>>>>>>> gp-master
        prev_filtered = clip.frame_buffer.get_last_filtered()
        # if prev_filtered is not None:
        #     delta = backsub - prev_filtered
        #     delta, _ = normalize(delta, new_max=255)
        #     cv2.imshow("delta", np.uint8(delta))
        clip.add_frame(thermal, backsub, saliencyMap, ffc_affected)
        f = clip.frame_buffer.get_last_frame()
        if clip.from_metadata:
            for track in clip.tracks:
                if clip.current_frame in track.frame_list:
                    track.add_frame_for_existing_region(
                        clip.frame_buffer.get_last_frame(),
                        threshold,
                        prev_filtered,
                    )
        else:
            regions = []
            if ffc_affected:
                clip.active_tracks = set()
            else:
                regions = self._get_regions_of_interest(
                    clip, component_details, backsub, prev_filtered, saliencyMap
                )
                self._apply_region_matchings(clip, regions, f)

            clip.region_history.append(regions)
<<<<<<< HEAD

    def _apply_region_matchings(self, clip, regions, f):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(
            clip, regions, f
        )
        new_tracks = self._create_new_tracks(clip, unmatched_regions, f)
        self._filter_inactive_tracks(clip, new_tracks, matched_tracks)

    def _match_existing_tracks(self, clip, regions, frame):
        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        matched_tracks = set()

        # logging.info("checking region %s", clip.current_frame)
        # for r in regions:
        # logging.info("%s", r)
        for track in clip.active_tracks:
            if track.stable:
                # using opencv tracker
                track.add_frame(frame)
                logging.info(
                    "%s adding frame to %s %s ",
                    clip.current_frame,
                    track,
                    track.last_bound,
                )
                matched_tracks.add(track)
                delete = []
                for region in regions:
                    overlap = track.last_bound.overlap_area(region) / region.area
                    if overlap > 0.8:
                        delete.append(region)

                for d in delete:
                    used_regions.add(d)
                    unmatched_regions.remove(d)
                    regions.remove(d)

                continue

            avg_mass = track.average_mass()
            max_distances = get_max_distance_change(track)
            for region in regions:
                distances, size_change = get_region_score(track.last_bound, region)
                logging.debug(
                    "checking region %s against %s %s", region, track, track.last_bound
                )
                max_size_change = get_max_size_change(track, region)
                max_mass_change = get_max_mass_change_percent(track, avg_mass)
                max_mass_change = None
                if max_mass_change and abs(avg_mass - region.mass) > max_mass_change:
                    self.print_if_verbose(
                        "track {} region mass {} deviates too much from {}".format(
                            track.get_id(),
                            region.mass,
                            avg_mass,
                        )
                    )

                    continue
                if distances[0] > max_distances[0]:
                    self.print_if_verbose(
                        "track {} distance score {} bigger than max distance {}".format(
                            track.get_id(), distances[0], max_distances[0]
                        )
                    )

                    continue
                elif max_distances[1] and distances[1] > max_distances[1]:
                    self.print_if_verbose(
                        "track {} mid point change score {} bigger than max distance {}".format(
                            track.get_id(), distances[1], max_distance[1]
                        )
                    )

                    continue
                elif max_distances[2] and distances[2] > max_distances[2]:
                    self.print_if_verbose(
                        "track {} width height change score {} bigger than max distance {}".format(
                            track.get_id(), distances[2], max_distance[2]
                        )
                    )

                    continue
                if size_change > max_size_change:
                    self.print_if_verbose(
                        "track {} size_change {} bigger than max size_change {}".format(
                            track.get_id(), size_change, max_size_change
                        )
                    )
                    continue
                scores.append((np.average(distances), track, region))

        # makes tracking consistent by ordering by score then by frame since target then track id
        scores.sort(
            key=lambda record: record[1].frames_since_target_seen
            + float(".{}".format(record[1]._id))
        )
        scores.sort(key=lambda record: record[0])
        for (score, track, region) in scores:
            if track in matched_tracks or region in used_regions:
                continue
            track.add_region(region, frame)
            logging.debug(
                "%s matched region %s to track %s %s",
                clip.current_frame,
                region,
                track,
                track.last_bound,
            )
            matched_tracks.add(track)
            used_regions.add(region)
            unmatched_regions.remove(region)

        return unmatched_regions, matched_tracks

    def _create_new_tracks(self, clip, unmatched_regions, f):
        """Create new tracks for any unmatched regions"""
        new_tracks = set()
        for region in unmatched_regions:
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in clip.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track.from_region(clip, region, ClipTrackExtractor.VERSION, f)
            new_tracks.add(track)
            clip._add_active_track(track)
            self.print_if_verbose(
                "Creating a new track {} with region {} mass{} area {} frame {}".format(
                    track.get_id(),
                    region,
                    track.last_bound.mass,
                    track.last_bound.area,
                    region.frame_number,
                )
            )
        return new_tracks

    def _filter_inactive_tracks(self, clip, new_tracks, matched_tracks):
        """Filters tracks which are or have become inactive"""

        unactive_tracks = clip.active_tracks
        # - matched_tracks - new_tracks
        clip.active_tracks = set()
        # matched_tracks | new_tracks
        for track in unactive_tracks:
            # need sufficient frames to allow insertion of excess blanks
            remove_after = min(
                2 * (len(track) - track.blank_frames),
                self.config.remove_track_after_frames,
            )
            if track.frames_since_target_seen + 1 < remove_after:
                if track.prev_frame_num != clip.current_frame:
                    track.add_blank_frame(clip.frame_buffer)
                    self.print_if_verbose(
                        "frame {} adding a blank frame to {} ".format(
                            clip.current_frame, track.get_id()
                        )
                    )
                clip.active_tracks.add(track)
            else:
                self.print_if_verbose(f"stop tracking track {track}")

    def _get_regions_of_interest(
        self, clip, component_details, filtered, prev_filtered, saliency
    ):
        """
                Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
                rectangles.
                :param filtered: The filtered frame
        =        :return: regions of interest, mask frame
        """

        if prev_filtered is not None:
            delta_frame = np.abs(filtered - prev_filtered)
        else:
            delta_frame = None

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        # find regions of interest
        regions = []
        for i, component in enumerate(component_details):
            # print("got component", component)

            if component[2] < 30 or component[3] < 30:
                continue
            # print("got component", component)

            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                mass=component[4],
                id=i,
                frame_number=clip.current_frame,
            )
            # if doing on both skip when no saliency in filtered
            # saliency_filtered = region.subimage(saliency)
            # num_pixels = len(saliency_filtered[saliency_filtered > 0])
            # print("num saliency", num_pixels, clip.current_frame, np.amax(saliency))
            # if num_pixels <= 4:
            #     print("skipped cause no saliency", clip.current_frame)
            #     continue
            # print("using cause saliency", clip.current_frame)

            old_region = region.copy()
            region.crop(clip.crop_rectangle)
            region.was_cropped = str(old_region) != str(region)
            region.set_is_along_border(clip.crop_rectangle)

            if self.config.cropped_regions_strategy == "cautious":
                crop_width_fraction = (
                    old_region.width - region.width
                ) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height
                ) / old_region.height
                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue
            elif (
                self.config.cropped_regions_strategy == "none"
                or self.config.cropped_regions_strategy is None
            ):
                if region.was_cropped:
                    continue
            elif self.config.cropped_regions_strategy != "all":
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy
                    )
                )
            region.enlarge(padding, max=clip.crop_rectangle)

            if delta_frame is not None:
                region_difference = region.subimage(delta_frame)
                region.pixel_variance = np.var(region_difference)

            # filter out regions that are probably just noise
            if (
                region.pixel_variance < self.config.aoi_pixel_variance
                and region.mass < self.config.aoi_min_mass
            ):
                continue
            regions.append(region)
        return regions

    def filter_tracks(self, clip):
        for track in clip.tracks:
            track.trim()
            track.set_end_s(clip.frames_per_second)

        track_stats = [(track.get_stats(), track) for track in clip.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)
        if self.verbose:
            for stats, track in track_stats:
                start_s, end_s = clip.start_and_end_in_secs(track)
                logging.info(
                    " - track %s duration: %.1fsec, number of frames:%s, stats %s",
                    track.get_id(),
                    end_s - start_s,
                    len(track),
                    stats,
                )
        # filter out tracks that probably are just noise.
        good_tracks = []
        self.print_if_verbose(
            "{} {}".format("Number of tracks before filtering", len(clip.tracks))
        )

        for stats, track in track_stats:
            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if not self.filter_track(clip, track, stats):
                good_tracks.append(track)

        clip.tracks = good_tracks
        self.print_if_verbose(
            "{} {}".format("Number of 'good' tracks", len(clip.tracks))
        )
        # apply max_tracks filter
        # note, we take the n best tracks.
        # if self.max_tracks is not None and self.max_tracks < len(clip.tracks):
        #     logging.warning(
        #         " -using only {0} tracks out of {1}".format(
        #             self.max_tracks, len(clip.tracks)
        #         )
        #     )
        #     clip.filtered_tracks.extend(
        #         [("Too many tracks", track) for track in clip.tracks[self.max_tracks :]]
        #     )
        #     clip.tracks = clip.tracks[: self.max_tracks]
        #
        # for key in clip.filtered_tracks:
        # self.print_if_verbose(
        #     "filtered track {} because {}".format(key[1].get_id(), key[0])
        # )

    def filter_track(self, clip, track, stats):
        # return not track.stable
        # return False
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * clip.frames_per_second:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            # return True
        #
        # # discard tracks that do not move enough
        #
        if (
            stats.max_offset < self.config.track_min_offset
            or stats.frames_moved < self.config.min_moving_frames
        ):
            self.print_if_verbose(
                "Track filtered.  Didn't move {}".format(stats.max_offset)
            )
            clip.filtered_tracks.append(("Track filtered.  Didn't move", track))
            return True

        if stats.blank_percent > self.config.max_blank_percent:
            print("blank percent", stats.blank_percent)
            self.print_if_verbose("Track filtered.  Too Many Blanks")
            clip.filtered_tracks.append(("Track filtered. Too Many Blanks", track))
            return True
        # if stats.region_jitter > self.config.max_jitter:
        #     self.print_if_verbose("Track filtered.  Too Jittery")
        #     clip.filtered_tracks.append(("Track filtered.  Too Jittery", track))
        #     return True
        # # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
        # if stats.delta_std < clip.track_min_delta:
        #     self.print_if_verbose(
        #         "Track filtered.  Too static {}".format(stats.delta_std)
        #     )
        #     clip.filtered_tracks.append(("Track filtered.  Too static", track))
        #     return True
        # if stats.delta_std > clip.track_max_delta:
        #     self.print_if_verbose("Track filtered.  Too Dynamic")
        #     clip.filtered_tracks.append(("Track filtered.  Too Dynamic", track))
        #     return True
        # # discard tracks that do not have enough enough average mass.
        # if stats.average_mass < self.config.track_min_mass:
        #     self.print_if_verbose(
        #         "Track filtered.  Mass too small ({})".format(stats.average_mass)
        #     )
        #     clip.filtered_tracks.append(("Track filtered.  Mass too small", track))
        #
        #     return True

        highest_ratio = 0
        for other in clip.tracks:
            if track == other:
                continue
            highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)

        if highest_ratio > self.config.track_overlap_ratio:
            self.print_if_verbose(
                "Track filtered.  Too much overlap {}".format(highest_ratio)
            )
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        return False

    def print_if_verbose(self, info_string):

        if self.verbose:
            logging.info(info_string)


def get_max_size_change(track, region):
    exiting = region.is_along_border and not track.last_bound.is_along_border
    entering = not exiting and track.last_bound.is_along_border
    region_percent = 1.5
    if len(track) < 5:
        # may increase at first
        region_percent = 2
    if entering or exiting:
        region_percent = 2

    return region_percent


def get_max_mass_change_percent(track, average_mass):
    if len(track) > ClipTrackExtractor.RESTRICT_MASS_AFTER * track.fps:
        vel = track.velocity
        mass_percent = ClipTrackExtractor.MASS_CHANGE_PERCENT
        if np.sum(np.abs(vel)) > 5:
            # faster tracks can be a bit more deviant
            mass_percent = mass_percent + 0.1
        return max(
            ClipTrackExtractor.MIN_MASS_CHANGE,
            average_mass * mass_percent,
        )
    else:
        return None


def get_max_distance_change(track):
    x, y = track.velocity
    x = max(x, 2)
    y = max(y, 2)
    if len(track) == 1:
        # be nice
        x = 10
        y = 10
    x = 8 * x
    y = 8 * y
    velocity_distance = x * x + y * y

    pred_vel = track.predicted_velocity()
    logging.debug(
        "%s velo %s pred vel %s", track, track.velocity, track.predicted_velocity()
    )
    pred_distance = pred_vel[0] * pred_vel[0] + pred_vel[1] * pred_vel[1]
    point_change = max(velocity_distance, pred_distance)
    distances = [velocity_distance, None, None]
    return distances
    #
    #
    # max_distance = ClipTrackExtractor.BASE_DISTANCE_CHANGE + max(
    #     velocity_distance, pred_distance
    # )
    # if max_distance > ClipTrackExtractor.MAX_DISTANCE:
    #     return ClipTrackExtractor.MAX_DISTANCE
    # return max_distance


def get_region_score(last_bound: Region, region: Region):
    """
    Calculates a score between 2 regions based of distance and area.
    The higher the score the more similar the Regions are
    """
    distances = last_bound.average_distance(region)

    # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
    # area is padded with 50 pixels so small regions don't change too much
    size_difference = abs(region.area - last_bound.area) / (last_bound.area + 50)

    return distances, size_difference


def get_filtered_frame(background, thermal):
    """
    Calculates filtered frame from thermal
    :param thermal: the thermal frame
    :param background: (optional) used for background subtraction
    :return: uint8 filtered frame and adjusted clip threshold for normalized frame
    """

    filtered = np.float32(thermal.copy())

    avg_change = 0
    filtered = filtered - background
    filtered[filtered < 0] = 0
    filtered, stats = normalize(filtered, new_max=255)

    # filtered[filtered > 10] += 30
    return filtered, 0
=======
>>>>>>> gp-master
