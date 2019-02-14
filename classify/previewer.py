
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import classify.globals as globs
from ml_tools import tools
from ml_tools.mpeg_creator import MPEGCreator
from track.trackextractor import TrackExtractor
from track.region import Region

HERE = os.path.dirname(os.path.dirname(__file__))
RESOURCES_PATH = os.path.join(HERE, "resources")

def resource_path(name):
    return os.path.join(RESOURCES_PATH, name)

class Previewer:

    def __init__(self, config):
        self.config = config
        # make sure all the required files are there
        self.track_descs = {}
        self.colourmap
        self.font
        self.font_title

    @property
    def colourmap(self):
        """ gets colourmap. """
        if not globs._previewer_colour_map:
            colourmap = self.config.previews_colour_map
            if os.path.exists(colourmap):
                print("loading colour map " + colourmap)
                self.colormap = tools.load_colormap(colourmap)
            else:
                print("using default colour map")
                self.colormap = plt.get_cmap('jet')

        return globs._previewer_colour_map


    @property
    def font(self):
        """ gets default font. """
        if not globs._previewer_font: globs._previewer_font = ImageFont.truetype(resource_path("Ubuntu-R.ttf"), 12)
        return globs._previewer_font

    @property
    def font_title(self):
        """ gets default title font. """
        if not globs._previewer_font_title: globs._previewer_font_title = ImageFont.truetype(resource_path("Ubuntu-B.ttf"), 14)
        return globs._previewer_font_title


    def export_clip_preview(self, filename, tracker:TrackExtractor, track_predictions):
        """
        Exports a clip showing the tracking and predictions for objects within the clip.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 4.0

        if tracker.stats:
            auto_max = tracker.stats['max_temp']
            auto_min = tracker.stats['min_temp']
            print("Using temperatures {}-{}".format(auto_min, auto_max))
        else:
            print("Do not have temperatures to use")
            return

        if track_predictions:
            self.create_track_descriptions(tracker, track_predictions)

        # setting quality to 30 gives files approximately the same size as the original CPTV MPEG previews
        # (but they look quite compressed)
        mpeg = MPEGCreator(filename)

        for frame_number, thermal in enumerate(tracker.frame_buffer.thermal):
            thermal_image = tools.convert_heat_to_img(thermal, self.colormap, auto_min, auto_max)
            thermal_image = thermal_image.resize((int(thermal_image.width * FRAME_SCALE), int(thermal_image.height * FRAME_SCALE)), Image.BILINEAR)

            if tracker.frame_buffer.filtered:
                # # if self.enable_side_by_side:
                # #     # put thermal & tracking images side by side
                # #     tracking_image = self.export_tracking_frame(tracker, frame_number, FRAME_SCALE, track_predictions)
                # #     side_by_side_image = Image.new('RGB', (tracking_image.width * 2, tracking_image.height))
                # #     side_by_side_image.paste(thermal_image, (0, 0))
                # #     side_by_side_image.paste(tracking_image, (tracking_image.width, 0))
                # #     mpeg.next_frame(np.asarray(side_by_side_image))
                # else:
                # overlay track rectanges on original thermal image
                thermal_image = self.draw_track_rectangles(tracker, frame_number, FRAME_SCALE, thermal_image, track_predictions)
                mpeg.next_frame(np.asarray(thermal_image))

            else:
                # no filtered frames available (clip too hot or
                # background moving?) so just output the original
                # frame without the tracking frame.
                mpeg.next_frame(np.asarray(thermal_image))

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break
        mpeg.close()

    def export_tracking_frame(self, tracker: TrackExtractor, frame_number:int, frame_scale:float, track_predictions):

        filtered = tracker.frame_buffer.filtered[frame_number]
        tracking_image = tools.convert_heat_to_img(filtered / 200, self.colormap, temp_min=0, temp_max=1)

        tracking_image = tracking_image.resize((int(tracking_image.width * frame_scale), int(tracking_image.height * frame_scale)), Image.NEAREST)

        return self.draw_track_rectangles(tracker, frame_number, frame_scale, tracking_image, track_predictions)

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


    def draw_track_rectangles(self, tracker, frame_number, frame_scale, image, track_predictions):
        draw = ImageDraw.Draw(image)

        # look for any tracks that occur on this frame
        for _, track in enumerate(tracker.tracks):

            frame_offset = frame_number - track.start_frame
            if 0 < frame_offset < len(track.bounds_history) - 1:
                # display the track
                rect = track.bounds_history[frame_offset].copy()

                rect_points = [int(p * frame_scale) for p in [rect.left, rect.top, rect.right, rect.top, rect.right,
                                                              rect.bottom, rect.left, rect.bottom, rect.left,
                                                              rect.top]]
                draw.line(rect_points, (255, 64, 32))

                if track_predictions:
                    prediction = track_predictions[track]

                    if track not in track_predictions:
                        # no information for this track just ignore
                        current_prediction_string = ''
                    else:
                        label = globs._classifier.labels[prediction.label_at_time(frame_offset)]
                        score = prediction.score_at_time(frame_offset)
                        if score >= 0.7:
                            prediction_format = "({:.1f} {})"
                        else:
                            prediction_format = "({:.1f} {})?"
                        current_prediction_string = prediction_format.format(score * 10, label)

                        current_prediction_string += "\nnovelty={:.2f}".format(prediction.novelty_history[frame_offset])

                    header_size = self.font_title.getsize(self.track_descs[track])
                    footer_size = self.font.getsize(current_prediction_string)

                    # figure out where to draw everything
                    header_rect = Region(rect.left * frame_scale, rect.top * frame_scale - header_size[1], header_size[0], header_size[1])
                    footer_center = ((rect.width * frame_scale) - footer_size[0]) / 2
                    footer_rect = Region(rect.left * frame_scale + footer_center, rect.bottom * frame_scale, footer_size[0], footer_size[1])

                    screen_bounds = Region(0, 0, image.width, image.height)

                    self.fit_to_screen(header_rect, screen_bounds)
                    self.fit_to_screen(footer_rect, screen_bounds)

                    draw.text((header_rect.x, header_rect.y), self.track_descs[track], font=self.font_title)
                    draw.text((footer_rect.x, footer_rect.y), current_prediction_string, font=self.font)

        return image

    def fit_to_screen(self, rect:Region, screen_bounds:Region):
        """ Modifies rect so that rect is visible within bounds. """
        if rect.left < screen_bounds.left:
            rect.x = screen_bounds.left
        if rect.top < screen_bounds.top:
            rect.y = screen_bounds.top

        if rect.right > screen_bounds.right:
            rect.x = screen_bounds.right - rect.width

        if rect.bottom > screen_bounds.bottom:
            rect.y = screen_bounds.bottom - rect.height
