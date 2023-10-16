import time
import os
from track.clip import Clip
from track.cliptrackextractor import ClipTrackExtractor
from config.config import Config
from ml_tools.previewer import Previewer


class TestTrackingSpeed:
    CPTV_FILE_NO_BACKGROUND = "clips/hedgehog.cptv"
    CPTV_FILE_BACKGROUND = "clips/possum.cptv"

    MAX_FRAME_MS = 40

    def test_tracking_speed(self):
        config = Config.get_defaults()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(dir_name, TestTrackingSpeed.CPTV_FILE_NO_BACKGROUND)
        print("Tracking cptv with no background ", file_name)
        track_extractor = ClipTrackExtractor(
            config.tracking,
            config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            cache_to_disk=False,
            verbose=config.verbose,
        )
        start = time.time()
        clip = Clip(config.tracking["thermal"], file_name)
        track_extractor.parse_clip(clip)
        ms_per_frame = (
            (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
        )
        print("Took {:.1f}ms per frame".format(ms_per_frame))
        assert ms_per_frame < TestTrackingSpeed.MAX_FRAME_MS

        file_name = os.path.join(dir_name, TestTrackingSpeed.CPTV_FILE_BACKGROUND)
        print("Tracking cptv with background ", file_name)
        start = time.time()
        clip = Clip(config.tracking["thermal"], file_name)
        track_extractor.parse_clip(clip)
        ms_per_frame = (
            (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
        )
        print("Took {:.1f}ms per frame".format(ms_per_frame))
        assert ms_per_frame < TestTrackingSpeed.MAX_FRAME_MS
