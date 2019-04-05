import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from ml_tools import tools
import ml_tools.globals as globs
from ml_tools.mpeg_creator import MPEGCreator
from track.trackextractor import TrackExtractor
from track.region import Region

HERE = os.path.dirname(os.path.dirname(__file__))
RESOURCES_PATH = os.path.join(HERE, "resources")

def resource_path(name):
    return os.path.join(RESOURCES_PATH, name)

class Previewer:

    PREVIEW_RAW = "raw"

    PREVIEW_CLASSIFIED = "classified"

    PREVIEW_NONE = "none"

    PREVIEW_TRACKING = "tracking"

    PREVIEW_BOXES = "boxes"

    PREVIEW_OPTIONS = [PREVIEW_NONE, PREVIEW_RAW, PREVIEW_CLASSIFIED, PREVIEW_TRACKING, PREVIEW_BOXES]

    TRACK_COLOURS = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 255, 255)]

    def __init__(self, config, preview_type):
        self.config = config
        # make sure all the required files are there
        self.track_descs = {}
        self.colourmap
        self.font
        self.font_title
        self.preview_type = preview_type
        self.frame_scale = 1

    @classmethod
    def create_if_required(self, config, preview_type):
        if not preview_type == Previewer.PREVIEW_NONE:
            return Previewer(config, preview_type)

    @property
    def colourmap(self):
        """ gets colourmap. """
        if not globs._previewer_colour_map:
            colourmap = self.config.previews_colour_map
            if os.path.exists(colourmap):
                self.colormap = tools.load_colormap(colourmap)
            else:
                print("using default colour map")
                self.colormap = plt.get_cmap('jet')

        return globs._previewer_colour_map


    @property
    def font(self):
        """ gets default font. """
        if not globs._previewer_font:
            globs._previewer_font = ImageFont.truetype(resource_path("Ubuntu-R.ttf"), 12)
        return globs._previewer_font

    @property
    def font_title(self):
        """ gets default title font. """
        if not globs._previewer_font_title:
            globs._previewer_font_title = ImageFont.truetype(resource_path("Ubuntu-B.ttf"), 14)
        return globs._previewer_font_title


    def export_clip_preview(self, filename, tracker: TrackExtractor, track_predictions=None):
        """
        Exports a clip showing the tracking and predictions for objects within the clip.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        if tracker.stats:
            self.auto_max = tracker.stats['max_temp']
            self.auto_min = tracker.stats['min_temp']
        else:
            print("Do not have temperatures to use.")
            return

        if bool(track_predictions) and self.preview_type == self.PREVIEW_CLASSIFIED:
            self.create_track_descriptions(tracker, track_predictions)

        if self.preview_type == self.PREVIEW_TRACKING and not tracker.frame_buffer.flow:
            tracker.generate_optical_flow()

        mpeg = MPEGCreator(filename)

        for frame_number, thermal in enumerate(tracker.frame_buffer.thermal):
            if self.preview_type == self.PREVIEW_RAW:
                image = self.convert_and_resize(thermal)

            if self.preview_type == self.PREVIEW_TRACKING:
                image = self.create_four_tracking_image(tracker.frame_buffer, frame_number)
                image = self.convert_and_resize(image, 3.0, mode=Image.NEAREST)
                draw = ImageDraw.Draw(image)
                regions = tracker.region_history[frame_number]
                self.add_regions(draw, regions)
                self.add_regions(draw, regions, v_offset=120)
                self.add_tracks(draw, tracker.tracks, frame_number)

            if self.preview_type == self.PREVIEW_BOXES:
                image = self.convert_and_resize(thermal, 4.0)
                draw = ImageDraw.Draw(image)
                screen_bounds = Region(0, 0, image.width, image.height)
                self.add_tracks(draw, tracker.tracks, frame_number)

            if self.preview_type == self.PREVIEW_CLASSIFIED:
                image = self.convert_and_resize(thermal, 4.0)
                draw = ImageDraw.Draw(image)
                screen_bounds = Region(0, 0, image.width, image.height)
                self.add_tracks(draw, tracker.tracks, frame_number, track_predictions, screen_bounds)

            mpeg.next_frame(np.asarray(image))

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break
        mpeg.close()

    def create_individual_track_previews(self, filename, tracker:TrackExtractor):
        # resolution of video file.
        # videos look much better scaled up
        filename_format = os.path.splitext(filename)[0] + "-{}.mp4"

        FRAME_SIZE = 4*48
        frame_width, frame_height = FRAME_SIZE, FRAME_SIZE
        for id, track in enumerate(tracker.tracks):
            video_frames = []
            for frame_number in range(len(track.bounds_history)):
                channels = tracker.get_track_channels(track, frame_number)
                img = tools.convert_heat_to_img(channels[1], self.colormap, 0, 350)
                img = img.resize((frame_width, frame_height), Image.NEAREST)
                video_frames.append(np.asarray(img))

            print("creating preview {}".format(filename_format.format(id + 1)))
            tools.write_mpeg(filename_format.format(id + 1), video_frames)


    def convert_and_resize(self, image, size = None, mode = Image.BILINEAR):
        """ Converts the image to colour using colour map and resize """
        image = tools.convert_heat_to_img(image, self.colormap, self.auto_min, self.auto_max)
        if size:
            self.frame_scale = size
            image = image.resize((int(image.width * self.frame_scale), int(image.height * self.frame_scale)), mode)
        return image

    def create_track_descriptions(self, tracker, track_predictions):
        # look for any tracks that occur on this frame
        for _, track in enumerate(tracker.tracks):

            prediction = track_predictions[track]
            # find a track description, which is the final guess of what this class is.
            guesses = ["{} ({:.1f})".format(
                globs._classifier.labels[prediction.label(i)], prediction.score(i) * 10) for i in range(1, 4)
                if prediction.score(i) > 0.5]
            track_description = "\n".join(guesses)
            track_description.strip()
            self.track_descs[track] = track_description

    def create_four_tracking_image(self, frame_buffer, frame_number):
        thermal = frame_buffer.thermal[frame_number]
        filtered = frame_buffer.filtered[frame_number] + self.auto_min
        mask = frame_buffer.mask[frame_number] * 10000
        flow = frame_buffer.flow[frame_number]
        flow_magnitude = np.linalg.norm(np.float32(flow), ord=2, axis=2) / 4.0 + self.auto_min

        return np.hstack((np.vstack((thermal, mask)),
                            np.vstack((filtered, flow_magnitude))))


    def add_regions(self, draw, regions, v_offset= 0):
        for rect in regions:
            draw.rectangle(self.rect_points(rect, v_offset), outline=(128, 128, 128))

    def add_tracks(self, draw, tracks, frame_number, track_predictions=None, screen_bounds=None):
        # look for any tracks that occur on this frame
        for index, track in enumerate(tracks):
            frame_offset = frame_number - track.start_frame
            if frame_offset >= 0 and frame_offset < len(track.bounds_history) - 1:
                rect = track.bounds_history[frame_offset]
                draw.rectangle(self.rect_points(rect), outline=self.TRACK_COLOURS[index % len(self.TRACK_COLOURS)])
                if track_predictions:
                    self.add_class_results(draw, track, frame_offset, rect, track_predictions, screen_bounds)

    def add_class_results(self, draw, track, frame_offset, rect, track_predictions, screen_bounds):
        prediction = track_predictions[track]
        if track not in track_predictions:
            return

        label = globs._classifier.labels[prediction.label_at_time(frame_offset)]
        score = prediction.score_at_time(frame_offset) * 10
        novelty = prediction.novelty_history[frame_offset]
        prediction_format = "({:.1f} {})\nnovelty={:.2f}"
        current_prediction_string = prediction_format.format(score * 10, label, novelty)

        header_size = self.font_title.getsize(self.track_descs[track])
        footer_size = self.font.getsize(current_prediction_string)

        # figure out where to draw everything
        header_rect = Region(rect.left * self.frame_scale, rect.top * self.frame_scale - header_size[1], header_size[0], header_size[1])
        footer_center = ((rect.width * self.frame_scale) - footer_size[0]) / 2
        footer_rect = Region(rect.left * self.frame_scale + footer_center, rect.bottom * self.frame_scale, footer_size[0], footer_size[1])

        self.fit_to_image(header_rect, screen_bounds)
        self.fit_to_image(footer_rect, screen_bounds)

        draw.text((header_rect.x, header_rect.y), self.track_descs[track], font=self.font_title)
        draw.text((footer_rect.x, footer_rect.y), current_prediction_string, font=self.font)

    def fit_to_image(self, rect:Region, screen_bounds:Region):
        """ Modifies rect so that rect is visible within bounds. """
        if rect.left < screen_bounds.left:
            rect.x = screen_bounds.left
        if rect.top < screen_bounds.top:
            rect.y = screen_bounds.top

        if rect.right > screen_bounds.right:
            rect.x = screen_bounds.right - rect.width

        if rect.bottom > screen_bounds.bottom:
            rect.y = screen_bounds.bottom - rect.height

    def rect_points(self, rect, v_offset = 0, h_offset = 0):
        s = self.frame_scale
        return [
            s * (rect.left + h_offset),
            s * (rect.top + v_offset),
            s * (rect.right + h_offset) - 1,
            s * (rect.bottom + v_offset) - 1,
        ]
        return
