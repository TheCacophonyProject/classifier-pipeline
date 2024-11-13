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
        self["segment_types"] = self.segment_types
        self["multi_label"] = True
        self["diff_norm"] = self.diff_norm
        self["thermal_diff_norm"] = self.thermal_diff_norm

        self["smooth_predictions"] = self.smooth_predictions
        self["channels"] = self.channels

    @property
    def channels(self):
        return self.get(
            "channels", [TrackChannels.thermal.name, TrackChannels.filtered.name]
        )

    @property
    def output_dim(self):
        if self.use_movement:
            return (
                self.frame_size * self.square_width,
                self.frame_size * self.square_width,
                len(self.channels),
            )
        return (self.frame_size, self.frame_size, len(self.channels))

    @property
    def smooth_predictions(self):
        return self.get("smooth_predictions", True)

    @property
    def excluded_labels(self):
        return self.get("excluded_labels", None)

    @property
    def remapped_labels(self):
        return self.get("remapped_labels", None)

    @property
    def thermal_diff_norm(self):
        return self.get("thermal_diff_norm", False)

    @property
    def diff_norm(self):
        return self.get("diff_norm", True)

    @property
    def multi_label(self):
        return self.get("multi_label", True)

    @property
    def keep_aspect(self):
        return self.get("keep_aspect", False)

    @property
    def use_background_filtered(self):
        return self.get("use_background_filtered", True)

    @property
    def keep_edge(self):
        return self.get("keep_edge", True)

    @property
    def segment_width(self):
        return self.get("segment_width", 25 if self.use_segments else 1)

    @property
    def segment_types(self):
        segment_types = self.get("segment_type", [SegmentType.ALL_RANDOM])
        # convert string to enum type
        if isinstance(segment_types, str):
            # old metadata
            segment_types = [SegmentType[segment_types]]
        elif isinstance(segment_types[0], str):
            for i in range(len(segment_types)):
                segment_types[i] = SegmentType[segment_types[i]]
        return segment_types

    @property
    def mvm(self):
        return self.get("mvm", False)

    @property
    def mvm_forest(self):
        return self.get("mvm_forest", False)

    @property
    def model_name(self):
        return self.get("model_name", "wr-resnet")

    @property
    def dense_sizes(self):
        return self.get("dense_sizes", None)

    @property
    def label_smoothing(self):
        return self.get("label_smoothing", 0)

    @property
    def base_training(self):
        return self.get("base_training", True)

    @property
    def retrain_layer(self):
        return self.get("retrain_layer")

    @property
    def dropout(self):
        return self.get("dropout", 0.3)

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

    def set_use_segments(self, use_segments):
        self["use_segments"] = use_segments
        if use_segments:
            self["square_width"] = 5
        else:
            self["square_width"] = 1

    #
    # @property
    # def red_type(self):
    #     ft = self.get("red_type", FrameTypes.thermal_tiled.name)
    #     return FrameTypes[ft]
    #
    # @property
    # def green_type(self):
    #     ft = self.get("green_type", FrameTypes.thermal_tiled.name)
    #     return FrameTypes[ft]
    #
    # @property
    # def blue_type(self):
    #     ft = self.get("blue_type", FrameTypes.thermal_tiled.name)
    #     return FrameTypes[ft]
