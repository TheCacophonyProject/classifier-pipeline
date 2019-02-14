from PIL import Image, ImageDraw
import numpy as np

from track.trackextractor import TrackExtractor
from ml_tools import tools

class MPEGPreviewStreamer():
    """ Generates MPEG preview frames from a tracker """

    def __init__(self, tracker: TrackExtractor, colormap):
        """
        Initialise the MPEG streamer.  Requires tracker frame_buffer to be allocated, and will generate optical
        flow if required.
        :param tracker:
        """
        self.tracker = tracker
        self.colormap = colormap
        assert tracker.frame_buffer, 'tracker frame buffer must be allocated for MPEG previews'
        self.tracker.generate_optical_flow()
        self.current_frame = 0
        self.FRAME_SCALE = 3.0
        self.track_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 255, 255)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame >= len(self.tracker.frame_buffer):
            raise StopIteration

        thermal = self.tracker.frame_buffer.thermal[self.current_frame]
        filtered = self.tracker.frame_buffer.filtered[self.current_frame]
        mask = self.tracker.frame_buffer.mask[self.current_frame]
        flow = self.tracker.frame_buffer.flow[self.current_frame]
        regions = self.tracker.region_history[self.current_frame]

        # marked is an image with each pixel's value being the label, 0...n for n objects
        # I multiply it here, but really I should use a seperate color map for this.
        # maybe I could multiply it modulo, and offset by some amount?

        # This really should be using a pallete here, I multiply by 10000 to make sure the binary mask '1' values get set to the brightest color (which is about 4000)
        # here I map the flow magnitude [ranges in the single didgits) to a temperature in the display range.
        flow_magnitude = np.linalg.norm(np.float32(flow), ord=2, axis=2) / 4.0

        stacked = np.hstack((np.vstack((thermal, mask * 10000)),
                             np.vstack((3 * filtered + tools.TEMPERATURE_MIN, flow_magnitude + tools.TEMPERATURE_MIN))))

        img = tools.convert_heat_to_img(stacked, self.colormap, tools.TEMPERATURE_MIN, tools.TEMPERATURE_MAX)
        img = img.resize((int(img.width * self.FRAME_SCALE), int(img.height * self.FRAME_SCALE)), Image.NEAREST)
        draw = ImageDraw.Draw(img)

        # look for any regions of interest that occur on this frame
        for rect in regions:
            rect_points = [int(p * self.FRAME_SCALE) for p in
                           [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom,
                            rect.left, rect.top]]
            draw.line(rect_points, (128, 128, 128))

        # look for any tracks that occur on this frame
        for id, track in enumerate(self.tracker.tracks):

            frame_offset = self.current_frame - track.start_frame
            if frame_offset >= 0 and frame_offset < len(track.bounds_history) - 1:
                # display the track
                rect = track.bounds_history[frame_offset]

                rect_points = [int(p * self.FRAME_SCALE) for p in
                               [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left,
                                rect.bottom, rect.left, rect.top]]
                draw.line(rect_points, self.track_colors[id % len(self.track_colors)])

        self.current_frame += 1

        return np.asarray(img)