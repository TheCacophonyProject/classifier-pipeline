from ml_tools.previewer import Previewer
from kalman.kalmanpreviewer import KalmanPreviewer


class KalmanPredictor:
    def __init__(self, config, base_filename, tracker):
        """Create an instance of a clip classifier"""

        self.base_filename = base_filename

        self.tracker = tracker

        self.previewer = KalmanPreviewer(config, Previewer.PREVIEW_BOXES)

    def make_preview(self):
        mp4_name = self.base_filename + ".mp4"
        print("making predictive preview {}".format(mp4_name))
        self.previewer.export_clip_preview(mp4_name, self.tracker)
