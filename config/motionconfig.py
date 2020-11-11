import attr
import inspect
from config import config
from .defaultconfig import DefaultConfig, deep_copy_map_if_key_not_exist


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
            camera_thresholds=[ThresholdConfig.get_defaults()],
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
class ThresholdConfig(DefaultConfig):

    camera_model = attr.ib()
    temp_thresh = attr.ib()
    delta_thresh = attr.ib()
    default = attr.ib()
    min_temp_thresh = attr.ib()
    max_temp_thresh = attr.ib()

    @classmethod
    def load(cls, threshold):
        defaults = cls.get_defaults()
        deep_copy_map_if_key_not_exist(defaults.as_dict(), threshold)
        return cls(
            camera_model=threshold["camera_model"],
            temp_thresh=threshold["temp_thresh"],
            delta_thresh=threshold["delta_thresh"],
            default=threshold["default"],
            min_temp_thresh=threshold["min_temp_thresh"],
            max_temp_thresh=threshold["max_temp_thresh"],
        )

    def as_dict(self):
        return attr.asdict(self)

    @classmethod
    def get_defaults(cls):
        return cls(
            camera_model="lepton3",
            temp_thresh=2900,
            delta_thresh=20,
            default=False,
            min_temp_thresh=None,
            max_temp_thresh=None,
        )

    def validate(self):
        return True
