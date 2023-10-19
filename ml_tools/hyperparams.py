from ml_tools.frame import TrackChannels
from ml_tools.datasetstructures import SegmentType
from ml_tools.preprocess import FrameTypes


class HyperParams(dict):
    """Helper wrapper for dictionary to make accessing hyper parameters easier"""

    def __init__(self, *args):
        super(HyperParams, self).__init__(*args)

        self.insert_defaults()

    def insert_defaults(self):
        self["model_name"] = self.model_name
        self["dense_sizes"] = self.dense_sizes
        self["base_training"] = self.base_training
        self["retrain_layer"] = self.retrain_layer
        self["dropout"] = self.dropout
        self["learning_rate"] = self.learning_rate
        self["learning_rate_decay"] = self.learning_rate_decay
        self["use_movement"] = self.use_movement
        self["use_segments"] = self.use_segments
        self["square_width"] = self.square_width
        self["frame_size"] = self.frame_size
        self["segment_width"] = self.segment_width

        self["shuffle"] = self.shuffle
        self["channel"] = self.channel
        self["type"] = self.type
        self["segment_type"] = self.segment_type
        self["multi_label"] = False

    @property
    def output_dim(self):
        if self.use_movement:
            return (
                self.frame_size * self.square_width,
                self.frame_size * self.square_width,
                3,
            )
        return (self.frame_size, self.frame_size, 3)

    @property
    def multi_label(self):
        return self.get("multi_label", False)

    @property
    def keep_aspect(self):
        return self.get("keep_aspect", False)

    @property
    def use_background_filtered(self):
        return self.get("use_background_filtered", True)

    @property
    def keep_edge(self):
        return self.get("keep_edge", False)

    @property
    def segment_width(self):
        return self.get("segment_width", 25 if self.use_segments else 1)

    @property
    def segment_type(self):
        segment_type = self.get("segment_type", SegmentType.ALL_RANDOM.name)
        if isinstance(segment_type, str):
            return SegmentType[segment_type]
        else:
            return segment_type

    # Model hyper paramters
    @property
    def type(self):
        return self.get("type", 1)

    @property
    def mvm(self):
        return self.get("mvm", False)

    @property
    def mvm_forest(self):
        return self.get("mvm_forest", False)

    @property
    def model_name(self):
        return self.get("model_name", "resnetv2")

    @property
    def channel(self):
        return self.get("channel", TrackChannels.thermal)

    @property
    def dense_sizes(self):
        return self.get("dense_sizes", None)

    @property
    def label_smoothing(self):
        return self.get("label_smoothing", 0)

    @property
    def base_training(self):
        return self.get("base_training", False)

    @property
    def retrain_layer(self):
        return self.get("retrain_layer")

    @property
    def dropout(self):
        return self.get("dropout")

    @property
    def learning_rate(self):
        return self.get("learning_rate", 0.001)

    @property
    def learning_rate_decay(self):
        return self.get("learning_rate_decay", None)

    # Datageneration parameters
    @property
    def batch_size(self):
        return self.get("batch_size", 32)

    @property
    def lstm(self):
        return self.get("lstm", False)

    @property
    def use_movement(self):
        return self.get("use_movement", True)

    @property
    def use_segments(self):
        return self.get("use_segments", True)

    @property
    def square_width(self):
        default = 1
        if self.use_segments:
            default = 5
        return self.get("square_width", default)

    @property
    def frame_size(self):
        return self.get("frame_size", 32)

    @property
    def shuffle(self):
        return self.get("shuffle", True)

    @property
    def maximum_preload(self):
        return self.get("maximum_preload", 1000)

    @property
    def red_type(self):
        type = self.get("red_type", FrameTypes.thermal_tiled.name)
        return FrameTypes[type]

    @property
    def green_type(self):
        type = self.get("green_type", FrameTypes.filtered_tiled.name)
        return FrameTypes[type]

    @property
    def blue_type(self):
        type = self.get("blue_type", FrameTypes.filtered_tiled.name)
        return FrameTypes[type]
