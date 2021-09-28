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


import logging
from os import path
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from load.clip import Clip
from ml_tools import tools
import ml_tools.globals as globs
from ml_tools.mpeg_creator import MPEGCreator
from track.region import Region
from ml_tools.imageprocessing import normalize


class Previewer:

    PREVIEW_RAW = "raw"

    PREVIEW_CLASSIFIED = "classified"

    PREVIEW_NONE = "none"

    PREVIEW_TRACKING = "tracking"

    PREVIEW_BOXES = "boxes"

    PREVIEW_OPTIONS = [
        PREVIEW_NONE,
        PREVIEW_RAW,
        PREVIEW_CLASSIFIED,
        PREVIEW_TRACKING,
        PREVIEW_BOXES,
    ]

    TRACK_COLOURS = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 255, 255)]
    FILTERED_COLOURS = [(128, 128, 128)]

    def __init__(self, config, preview_type):
        self.config = config
        self.colourmap = self._load_colourmap()

        # make sure all the required files are there
        self.track_descs = {}
        self.preview_type = preview_type
        self.debug = config.debug

    @classmethod
    def create_if_required(self, config, preview_type):
        if not preview_type.lower() == Previewer.PREVIEW_NONE:
            return Previewer(config, preview_type)

    def _load_colourmap(self):
        colourmap_path = self.config.previews_colour_map
        if not path.exists(colourmap_path):
            colourmap_path = tools.resource_path("colourmap.dat")
        return tools.load_colourmap(colourmap_path)

    def export_clip_preview(self, filename, clip: Clip, predictions=None):
        """
        Exports a clip showing the tracking and predictions for objects within the clip.
        """

        logging.info("creating clip preview %s", filename)

        # increased resolution of video file.
        # videos look much better scaled up
        if not clip.stats:
            logging.error("Do not have temperatures to use.")
            return

        if self.debug:
            footer = Previewer.stats_footer(clip.stats)
        if predictions and (
            self.preview_type == self.PREVIEW_CLASSIFIED
            or self.preview_type == self.PREVIEW_TRACKING
        ):
            self.create_track_descriptions(clip, predictions)

        if clip.stats.min_temp is None or clip.stats.max_temp is None:
            thermals = [frame.thermal for frame in clip.frame_buffer.frames]
            clip.stats.min_temp = np.amin(thermals)
            clip.stats.max_temp = np.amax(thermals)
        mpeg = MPEGCreator(filename)
        frame_scale = 4.0
        for frame_number, frame in enumerate(clip.frame_buffer):
            if self.preview_type == self.PREVIEW_RAW:
                image = self.convert_and_resize(
                    frame.thermal, clip.stats.min_temp, clip.stats.max_temp
                )
                draw = ImageDraw.Draw(image)
            elif self.preview_type == self.PREVIEW_TRACKING:
                image = self.create_four_tracking_image(
                    frame,
                    clip.stats.min_temp,
                    clip.stats.max_temp,
                )
                draw = ImageDraw.Draw(image)
                self.add_tracks(draw, clip.tracks, frame_number, predictions, scale=4.0)

            elif self.preview_type == self.PREVIEW_BOXES:
                image = self.convert_and_resize(
                    frame.thermal, clip.stats.min_temp, clip.stats.max_temp
                )
                draw = ImageDraw.Draw(image)
                screen_bounds = Region(0, 0, image.width, image.height)
                self.add_tracks(
                    draw, clip.tracks, frame_number, colours=[(128, 255, 255)]
                )

            elif self.preview_type == self.PREVIEW_CLASSIFIED:
                image = self.convert_and_resize(
                    frame.thermal, clip.stats.min_temp, clip.stats.max_temp
                )
                draw = ImageDraw.Draw(image)
                screen_bounds = Region(0, 0, image.width, image.height)
                self.add_tracks(
                    draw, clip.tracks, frame_number, predictions, screen_bounds
                )
            if frame.ffc_affected:
                self.add_header(draw, image.width, image.height, "Calibrating ...")
            if self.debug and draw:
                self.add_footer(
                    draw, image.width, image.height, footer, frame.ffc_affected
                )
            mpeg.next_frame(np.asarray(image))

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > clip.frames_per_second * 60 * 10:
                break
        clip.frame_buffer.close_cache()
        mpeg.close()

    def create_individual_track_previews(self, filename, clip: Clip):
        # resolution of video file.
        # videos look much better scaled up
        filename_format = path.splitext(filename)[0] + "-{}.mp4"

        FRAME_SIZE = 4 * 48
        frame_width, frame_height = FRAME_SIZE, FRAME_SIZE

        for id, track in enumerate(clip.tracks):
            video_frames = []
            for region in track.bounds_history:
                frame = clip.frame_buffer.get_frame(region.frame_number)
                cropped = frame.crop_by_region(region)
                if cropped.thermal.size == 0:
                    continue
                img = tools.convert_heat_to_img(
                    cropped.thermal,
                    self.colourmap,
                    np.amin(cropped.thermal),
                    np.amax(cropped.thermal),
                )
                img = img.resize((frame_width, frame_height), Image.NEAREST)
                video_frames.append(np.asarray(img))

            logging.info("creating preview %s", filename_format.format(id + 1))
            tools.write_mpeg(filename_format.format(id + 1), video_frames)

    def convert_and_resize(self, frame, h_min, h_max, size=None, mode=Image.BILINEAR):
        """Converts the image to colour using colour map and resize"""
        thermal = frame[:120, :160].copy()
        image = tools.convert_heat_to_img(frame, self.colourmap, h_min, h_max)
        image = image.resize(
            (
                int(image.width * self.frame_scale),
                int(image.height * self.frame_scale),
            ),
            mode,
        )

        if self.debug:
            tools.add_heat_number(image, thermal, self.frame_scale)
        return image

    def create_track_descriptions(self, clip, predictions):
        if predictions is None:
            return
        # look for any tracks that occur on this frame
        for track in clip.tracks:
            guesses = predictions.guesses_for(track.get_id())
            track_description = "\n".join(guesses)
            track_description.strip()
            self.track_descs[track] = track_description

    def add_regions(draw, regions, v_offset=0, scale=1):
        for rect in regions:
            draw.rectangle(
                rect_points(rect, v_offset, scale=scale), outline=(128, 128, 128)
            )

    def add_tracks(
        self,
        draw,
        tracks,
        frame_number,
        track_predictions=None,
        screen_bounds=None,
        colours=TRACK_COLOURS,
        tracks_text=None,
        v_offset=0,
        scale=1,
    ):

        # look for any tracks that occur on this frame
        for index, track in enumerate(tracks):
            frame_offset = frame_number - track.start_frame
            if frame_offset >= 0 and frame_offset < len(track.bounds_history):
                region = track.bounds_history[frame_offset]
                prediction = None
                prediction = track_predictions.prediction_for(track.get_id())
                add_track(
                    draw,
                    track,
                    self.track_descs[track],
                    region,
                    frame_offset,
                    colours[index % len(colours)],
                    prediction,
                    screen_bounds,
                    tracks_text,
                    v_offset=v_offset,
                    debug=self.debug,
                    scale=scale,
                )

    def add_header(
        self,
        draw,
        width,
        height,
        text,
    ):
        font = get_font()
        footer_size = font.getsize(text)
        center = (width / 2 - footer_size[0] / 2.0, 5)
        draw.text((center[0], center[1]), text, font=font)

    def add_footer(self, draw, width, height, text, ffc_affected):
        footer_text = "FFC {} {}".format(ffc_affected, text)
        footer_size = font.getsize(footer_text)
        center = (width / 2 - footer_size[0] / 2.0, height - footer_size[1])
        draw.text((center[0], center[1]), footer_text, font=font)

    def add_debug_text(
        draw,
        track,
        region,
        screen_bounds,
        text=None,
        v_offset=0,
        frame_offset=0,
        scale=1,
    ):
        font = get_font()
        if text is None:
            text = "id {}".format(track.get_id())
            if region.pixel_variance:
                text += "mass {} var {} vel ({},{})".format(
                    region.mass,
                    round(region.pixel_variance, 2),
                    track.vel_x[frame_offset],
                    track.vel_y[frame_offset],
                )
        footer_size = font.getsize(text)
        footer_center = ((region.width * self.frame_scale) - footer_size[0]) / 2

        footer_rect = Region(
            region.right * scale - footer_center / 2.0,
            (v_offset + region.bottom) * self.frame_scale,
            footer_size[0],
            footer_size[1],
        )
        fit_to_image(footer_rect, screen_bounds)

        draw.text((footer_rect.x, footer_rect.y), text, font=font)

    def create_four_tracking_image(self, frame, min_temp, max_temp):

        thermal = frame.thermal
        filtered = frame.filtered + min_temp
        filtered = tools.convert_heat_to_img(
            filtered, self.colourmap, min_temp, max_temp
        )
        thermal = tools.convert_heat_to_img(thermal, self.colourmap, min_temp, max_temp)
        if self.debug:
            tools.add_heat_number(thermal, frame.thermal, 1)
        mask, _ = normalize(frame.mask, new_max=255)
        mask = np.uint8(mask)

        mask = mask[..., np.newaxis]
        mask = np.repeat(mask, 3, axis=2)

        mask = Image.fromarray(mask)
        flow_h, flow_v = frame.get_flow_split(clip_flow=True)
        if flow_h is None and flow_v is None:
            flow_magnitude = None
        else:
            flow_magnitude = (
                np.linalg.norm(np.float32([flow_h, flow_v]), ord=2, axis=0) / 4.0
                + min_temp
            )
            flow_magnitude = tools.convert_heat_to_img(
                flow_magnitude, self.colourmap, min_temp, max_temp
            )

        image = np.hstack(
            (np.vstack((thermal, mask)), np.vstack((filtered, flow_magnitude)))
        )
        image = Image.fromarray(image)

        image = image.resize(
            (
                int(image.width * 4),
                int(image.height * 4),
            ),
            Image.BILINEAR,
        )
        return image

    @staticmethod
    def stats_footer(stats):
        return "max {}, min{}, mean{}, filtered deviation {}, avg delta{}, temp_thresh {}".format(
            none_or_round(stats.max_temp),
            none_or_round(stats.min_temp),
            none_or_round(stats.mean_temp),
            none_or_round(stats.filtered_deviation, 2),
            none_or_round(stats.average_delta, 1),
            none_or_round(stats.temp_thresh),
        )


def none_or_round(value, decimals=0):
    if value:
        return round(value, decimals)
    else:
        return value


def add_class_results(
    draw,
    track,
    track_desc,
    frame_offset,
    rect,
    track_prediction,
    screen_bounds,
    v_offset=0,
    scale=1,
):

    if track_prediction is None:
        return

    current_prediction_string = track_prediction.get_classified_footer(frame_offset)
    add_text_to_track(
        draw,
        rect,
        track_desc,
        current_prediction_string,
        screen_bounds,
        v_offset,
        scale=scale,
    )


def add_text_to_track(
    draw, rect, header_text, footer_text, screen_bounds, v_offset=0, scale=1
):
    font = get_font()
    font_title = get_title_font()
    header_size = font_title.getsize(header_text)
    footer_size = font.getsize(footer_text)
    # figure out where to draw everything
    header_rect = Region(
        rect.left * scale,
        (v_offset + rect.top) * scale - header_size[1],
        header_size[0],
        header_size[1],
    )
    footer_center = ((rect.width * scale) - footer_size[0]) / 2
    footer_rect = Region(
        rect.left * scale + footer_center,
        (v_offset + rect.bottom) * scale,
        footer_size[0],
        footer_size[1],
    )

    fit_to_image(header_rect, screen_bounds)
    fit_to_image(footer_rect, screen_bounds)

    draw.text((header_rect.x, header_rect.y), header_text, font=font_title)
    draw.text((footer_rect.x, footer_rect.y), footer_text, font=font)


def fit_to_image(rect: Region, screen_bounds: Region):
    """Modifies rect so that rect is visible within bounds."""
    if screen_bounds is None:
        return
    if rect.left < screen_bounds.left:
        rect.x = screen_bounds.left
    if rect.top < screen_bounds.top:
        rect.y = screen_bounds.top

    if rect.right > screen_bounds.right:
        rect.x = screen_bounds.right - rect.width

    if rect.bottom > screen_bounds.bottom:
        rect.y = screen_bounds.bottom - rect.height


def add_track(
    draw,
    track,
    track_desc,
    region,
    frame_offset,
    colour,
    track_prediction=None,
    screen_bounds=None,
    tracks_text=None,
    v_offset=0,
    scale=1,
    debug=False,
):
    draw.rectangle(rect_points(region, v_offset, scale=scale), outline=colour)
    if track_prediction:
        add_class_results(
            draw,
            track,
            track_desc,
            frame_offset,
            region,
            track_prediction,
            screen_bounds,
            v_offset=v_offset,
            scale=scale,
        )
    if debug:
        text = None
        if tracks_text and len(tracks_text) > index:
            text = tracks_text[index]
        add_debug_text(
            draw,
            track,
            region,
            screen_bounds,
            text=text,
            v_offset=v_offset,
            frame_offset=frame_offset,
            scale=scale,
        )


def get_font():
    """gets default font."""
    if not globs._previewer_font:
        globs._previewer_font = ImageFont.truetype(
            tools.resource_path("Ubuntu-R.ttf"), 12
        )
    return globs._previewer_font


def get_title_font():
    """gets default title font."""
    if not globs._previewer_font_title:
        globs._previewer_font_title = ImageFont.truetype(
            tools.resource_path("Ubuntu-B.ttf"), 14
        )
    return globs._previewer_font_title


def rect_points(rect, v_offset=0, h_offset=0, scale=1):
    return [
        scale * (rect.left + h_offset),
        scale * (rect.top + v_offset),
        scale * (rect.right + h_offset) - 1,
        scale * (rect.bottom + v_offset) - 1,
    ]
    return


def add_last_frame_tracking(
    frame,
    tracks,
    track_predictions=None,
    screen_bounds=None,
    colours=Previewer.TRACK_COLOURS,
    tracks_text=None,
    v_offset=0,
    scale=1,
    debug=False,
):
    image = tools.convert_heat_to_img(frame.pix)

    draw = ImageDraw.Draw(image)

    # look for any tracks that occur on this frame
    for index, track in enumerate(tracks):
        region = track.bounds_history[-1]
        if region.frame_number != frame.frame_number:
            continue
        draw.rectangle(
            rect_points(region, v_offset, scale=scale),
            outline=colours[index % len(colours)],
        )
        track_prediction = track_predictions.prediction_for(track)
        if track_prediction:
            footer_text = track_prediction.get_classified_footer(labels)
            add_text_to_track(
                draw,
                region,
                str(track.get_id()),
                footer_text,
                screen_bounds,
                v_offset,
            )

        if debug:
            text = None
            if tracks_text and len(tracks_text) > index:
                text = tracks_text[index]
            add_debug_text(
                draw, track, region, screen_bounds, text=text, v_offset=v_offset
            )
