import attr
import inspect
from .defaultconfig import DefaultConfig


@attr.s
class MotionConfig(DefaultConfig):
    camera_thresholds = attr.ib()
    dynamic_thresh = attr.ib()

    @classmethod
    def load(cls, threshold):
        if inspect.isclass(threshold):
            return threshold
        return cls(
            camera_thresholds=MotionConfig.load_camera_thresholds(
                threshold.get("camera_thresholds")
            ),
            dynamic_thresh=threshold["dynamic_thresh"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            camera_thresholds={
                "max_temp_thresh": 2900,
                "temp_thresh": 2900,
                "background_thresh": 20,
                "defualt": True,
                "camera_model": "lepton3",
            },
            dynamic_thresh=True,
        )

    def load_camera_thresholds(raw):
        if raw is None:
            return None

        threholds = {}
        for raw_threshold in raw:
            threshold = ThresholdConfig.load(raw_threshold)
            threholds[threshold.camera_model] = threshold
        return threholds

    def validate(self):
        return True

    def as_dict(self):
        return attr.asdict(self)

    def threshold_for_model(self, camera_model):
        if self.camera_thresholds is None:
            return None
        threshold = self.camera_thresholds.get(camera_model)
        if not threshold:
            for model_thresh in self.camera_thresholds.values():
                if model_thresh.default:
                    threshold = model_thresh
                    break
        return threshold


@attr.s
class ThresholdConfig:

    camera_model = attr.ib()
    temp_thresh = attr.ib()
    background_thresh = attr.ib()
    default = attr.ib()
    min_temp_thresh = attr.ib()
    max_temp_thresh = attr.ib()
    track_min_delta = attr.ib()
    track_max_delta = attr.ib()

    @classmethod
    def load(cls, threshold):
        return cls(
            camera_model=threshold["camera_model"],
            temp_thresh=threshold["temp_thresh"],
            background_thresh=threshold.get("background_thresh", 20),
            default=threshold.get("default", False),
            min_temp_thresh=threshold.get("min_temp_thresh"),
            max_temp_thresh=threshold.get("max_temp_thresh"),
            track_min_delta=threshold.get("track_min_delta", 1),
            track_max_delta=threshold.get("track_max_delta", 150),
        )
