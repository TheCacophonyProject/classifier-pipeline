from abc import ABC, abstractmethod
import time
import json
import logging
import numpy as np
from ml_tools.hyperparams import HyperParams
from pathlib import Path
from classify.trackprediction import TrackPrediction
from ml_tools.imageprocessing import normalize


class Interpreter(ABC):
    def __init__(self, model_file):
        self.load_json(model_file)
        self.id = None

    def load_json(self, filename):
        """Loads model and parameters from file."""
        filename = Path(filename)
        filename = filename.with_suffix(".json")
        logging.info("Loading metadata from %s", filename)
        metadata = json.load(open(filename, "r"))
        self.version = metadata.get("version", None)
        self.labels = metadata["labels"]
        self.params = HyperParams()
        self.params["remapped_labels"] = metadata.get("remapped_labels")
        self.params["excluded_labels"] = metadata.get("excluded_labels")

        self.params.update(metadata.get("hyperparams", {}))
        self.data_type = metadata.get("type", "thermal")

        self.mapped_labels = metadata.get("mapped_labels")
        self.label_probabilities = metadata.get("label_probabilities")
        self.preprocess_fn = self.get_preprocess_fn()

    @abstractmethod
    def shape(self):
        """Num Inputs, Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def get_preprocess_fn(self):
        model_name = self.params.model_name
        if model_name == "inceptionv3":
            # no need to use tf module, if train other model types may have to add
            #  preprocess definitions
            return inc3_preprocess
        elif model_name == "wr-resnet":
            return None
        else:
            import tensorflow as tf

            if model_name == "resnet":
                return tf.keras.applications.resnet.preprocess_input
            elif model_name == "nasnet":
                return tf.keras.applications.nasnet.preprocess_input
            elif model_name == "resnetv2":
                return tf.keras.applications.resnet_v2.preprocess_input

            elif model_name == "resnet152":
                return tf.keras.applications.resnet.preprocess_input

            elif model_name == "vgg16":
                return tf.keras.applications.vgg16.preprocess_input

            elif model_name == "vgg19":
                return tf.keras.applications.vgg19.preprocess_input

            elif model_name == "mobilenet":
                return tf.keras.applications.mobilenet_v2.preprocess_input

            elif model_name == "densenet121":
                return tf.keras.applications.densenet.preprocess_input

            elif model_name == "inceptionresnetv2":
                return tf.keras.applications.inception_resnet_v2.preprocess_input
        logging.warn("pretrained model %s has no preprocessing function", model_name)
        return None

    def preprocess(self, clip, track, **args):
        scale = args.get("scale", None)
        num_predictions = args.get("num_predictions", None)
        predict_from_last = args.get("predict_from_last", None)
        segment_frames = args.get("segment_frames")
        frames_per_classify = args.get("frames_per_classify", 25)
        available_frames = (
            min(len(track.bounds_history), clip.frames_kept())
            if clip.frames_kept() is not None
            else len(track.bounds_history)
        )
        if predict_from_last is not None:
            predict_from_last = min(predict_from_last, available_frames)
        # this might be a little slower as it checks some massess etc
        # but keeps it the same for all ways of classifying
        if frames_per_classify > 1:
            if predict_from_last is not None and segment_frames is None:
                logging.debug(
                    "Prediction from last available frames %s track is of length %s",
                    available_frames,
                    len(track.bounds_history),
                )
                regions = track.bounds_history[-available_frames:]
                valid_regions = 0
                if available_frames > predict_from_last:
                    # want to get rid of any blank frames
                    predict_from_last = 0
                    for i, r in enumerate(
                        reversed(track.bounds_history[-available_frames:])
                    ):
                        if r.blank:
                            continue
                        valid_regions += 1
                        predict_from_last = i + 1
                        if valid_regions >= predict_from_last:
                            break
                logging.debug(
                    "After checking blanks have predict from last %s from last available frames %s track is of length %s",
                    predict_from_last,
                    available_frames,
                    len(track.bounds_history),
                )

            if args.get("do_contours", True):
                logging.info("DOING CONTOURS")
                # logging.warn("Implemented for testing")
                frames, preprocessed, masses = self.preprocess_contours(
                    clip,
                    track,
                )
            else:
                frames, preprocessed, masses = self.preprocess_segments(
                    clip,
                    track,
                    num_predictions,
                    predict_from_last,
                    segment_frames=segment_frames,
                    dont_filter=args.get("dont_filter", False),
                    min_segments=args.get("min_segments"),
                )
        else:
            frames, preprocessed, masses = self.preprocess_frames(
                clip, track, num_predictions, segment_frames=segment_frames
            )
        return frames, preprocessed, masses

    def classify_track(self, clip, track, segment_frames=None, min_segments=None):
        start = time.time()
        prediction_frames, output, masses = self.predict_track(
            clip,
            track,
            segment_frames=segment_frames,
            frames_per_classify=self.params.square_width**2,
            min_segments=min_segments,
        )
        if prediction_frames is None:
            return None
        mustelid_i = self.labels.index("mustelid")
        for i, pred in enumerate(output):
            a_sort = np.argsort(pred)
            for arg in a_sort[-2:]:
                label = self.labels[arg]
                conf = pred[arg]
                logging.info(
                    "%s label %s Got %s confidence", i + 1, label, round(conf * 100)
                )
            label = self.labels[mustelid_i]
            conf = pred[mustelid_i]
            logging.info(
                "%s label %s Got %s confidence", i + 1, label, round(conf * 100)
            )
        if output is None:
            logging.info("Skipping track %s", track.get_id())
            return None
        track_prediction = TrackPrediction(track.get_id(), self.labels)
        # self.model.predict(preprocessed)
        top_score = None
        smoothed_predictions = None

        if self.params.smooth_predictions:
            masses = np.array(masses)
            top_score = np.sum(masses)
            masses = masses[:, None]
            smoothed_predictions = output * masses
        track_prediction.classified_clip(
            output,
            smoothed_predictions,
            prediction_frames,
            masses,
            top_score=top_score,
        )
        if (
            len(prediction_frames) == 1
            and len(set(prediction_frames[0])) < self.params.square_width**2 / 4
        ):
            # if we don't have many frames to get a good prediction, lets assume only false-positive is a good prediction and filter the rest to a maximum of 0.5
            if track_prediction.predicted_tag() != "false-positive":
                track_prediction.cap_confidences(0.5)
        track_prediction.classify_time = time.time() - start
        return track_prediction

    def predict_track(self, clip, track, **args):
        frames, preprocessed, masses = self.preprocess(clip, track, **args)
        if preprocessed is None or len(preprocessed) == 0:
            return None, None, None
        pred = self.predict(preprocessed)
        return frames, pred, masses

    def preprocess_frames(
        self,
        clip,
        track,
        max_frames=None,
        segment_frames=None,
    ):
        from ml_tools.preprocess import preprocess_single_frame, preprocess_frame

        data = []
        frames_used = []
        filtered_norm_limits = None
        thermal_norm_limits = None
        if self.params.diff_norm or self.params.thermal_diff_norm:
            min_diff = None
            max_diff = 0
            thermal_max_diff = None
            thermal_min_diff = None
            for i, region in enumerate(reversed(track.bounds_history)):
                if region.blank:
                    continue
                if region.width == 0 or region.height == 0:
                    logging.warn(
                        "No width or height for frame %s regoin %s",
                        region.frame_number,
                        region,
                    )
                    continue
                f = clip.get_frame(region.frame_number)
                if region.blank or region.width <= 0 or region.height <= 0:
                    continue

                f.float_arrays()

                if self.params.thermal_diff_norm:
                    diff_frame = f.thermal - np.median(f.thermal)
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if thermal_min_diff is None or new_min < thermal_min_diff:
                        thermal_min_diff = new_min
                    if thermal_max_diff is None or new_max > thermal_max_diff:
                        thermal_max_diff = new_max
                if self.params.diff_norm:
                    diff_frame = region.subimage(f.filtered)
                    # region.subimage(f.thermal) - region.subimage(
                    # clip.background
                    # )

                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if min_diff is None or new_min < min_diff:
                        min_diff = new_min
                    if new_max > max_diff:
                        max_diff = new_max
            if self.params.thermal_diff_norm:
                thermal_norm_limits = (thermal_min_diff, thermal_max_diff)

            if self.params.diff_norm:
                filtered_norm_limits = (min_diff, max_diff)

        for i, region in enumerate(reversed(track.bounds_history)):
            if region.blank:
                continue
            if region.width == 0 or region.height == 0:
                logging.warn(
                    "No width or height for frame %s regoin %s",
                    region.frame_number,
                    region,
                )
                continue
            frame = clip.get_frame(region.frame_number)
            if frame is None:
                logging.error(
                    "Clasifying clip %s track %s can't get frame %s",
                    clip.get_id(),
                    track.get_id(),
                    region.frame_number,
                )
                raise Exception(
                    "Clasifying clip {} track {} can't get frame {}".format(
                        clip.get_id(), track.get_id(), region.frame_number
                    )
                )
            logging.debug(
                "classifying single frame with preprocess %s size %s crop? %s f shape %s region %s",
                "None" if self.preprocess_fn is None else self.preprocess_fn.__module__,
                self.params.frame_size,
                True,
                frame.thermal.shape,
                region,
            )
            cropped_frame = preprocess_frame(
                frame,
                (self.params.frame_size, self.params.frame_size),
                region,
                clip.background,
                clip.crop_rectangle,
                calculate_filtered=False,
                filtered_norm_limits=filtered_norm_limits,
                thermal_norm_limits=thermal_norm_limits,
            )
            preprocessed = preprocess_single_frame(
                cropped_frame,
                self.params.channels,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )

            frames_used.append(region.frame_number)
            data.append(preprocessed)
            if max_frames is not None and len(data) >= max_frames:
                break
        return frames_used, np.array(data), [region.mass]

    def preprocess_contours(self, clip, track):

        # should really be over whole track buts let just do the indices we predict of
        #  seems to make little different to just doing a min max normalization
        thermal_norm_limits = None
        filtered_norm_limits = None
        data = []
        weights = []
        indices = []
        prev_elongation = None
        valid_regions = False
        prev_region = None
        for region in track.bounds_history:
            if prev_region is not None:
                velo = np.array(region.mid) - np.array(prev_region.mid)
                print("Velo is ",velo, sum(abs(velo)))
                region.velocity = sum(abs(velo))
            else:
                region.velocity = 0
            # pred_vel_y = self.predicted_mid[1] - self.last_bound.centroid[1]
            prev_region = region
            region.set_is_along_border(clip.crop_rectangle, edge=1)
            if not region.is_along_border:
                valid_regions = True

        if self.params.diff_norm or self.params.thermal_diff_norm:
            min_diff = None
            max_diff = 0
            thermal_max_diff = None
            thermal_min_diff = None
            for i, region in enumerate(track.bounds_history):
                if region.blank:
                    continue
                if region.width <= 0 or region.height <= 0:
                    logging.warn(
                        "No width or height for frame %s regoin %s",
                        region.frame_number,
                        region,
                    )
                    continue
                f = clip.get_frame(region.frame_number)
                region.set_is_along_border(clip.crop_rectangle, edge=1)
                # logging.info("%s is along border %s %s elong %s prev %s",region.frame_number,region.is_along_border,region,region.elongation,prev_elongation)
                if region.blank or region.width <= 0 or region.height <= 0 or f is None:
                    # or region.is_along_border:
                    continue

                if region.is_along_border and valid_regions:
                    if (
                        prev_elongation is None
                        or abs(prev_elongation - region.elongation) > 0.15
                    ):
                        logging.info("ALong border skip %s",region.frame_number)
                        continue
                        # break
                else:
                    prev_elongation = region.elongation
                f.float_arrays()

                if self.params.thermal_diff_norm:
                    diff_frame = f.thermal - np.median(f.thermal)
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if thermal_min_diff is None or new_min < thermal_min_diff:
                        thermal_min_diff = new_min
                    if thermal_max_diff is None or new_max > thermal_max_diff:
                        thermal_max_diff = new_max
                if self.params.diff_norm:
                    diff_frame = region.subimage(f.filtered)
                    # - region.subimage(
                    #     clip.background
                    # )
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if min_diff is None or new_min < min_diff:
                        min_diff = new_min
                    if new_max > max_diff:
                        max_diff = new_max
                # copy = region.copy()
                # copy.enlarge(5, max=clip.crop_rectangle)
                # contours = get_contours(copy.subimage(f.filtered), f.frame_number)
                f.region = region
                # weights.append(contours)
                # logging.info("Frame %s contour %s", f.frame_number,contours)
                data.append(f)

            if self.params.thermal_diff_norm:
                thermal_norm_limits = (thermal_min_diff, thermal_max_diff)

            if self.params.diff_norm:
                filtered_norm_limits = (min_diff, max_diff)
            data = np.array(data)

            # take about half the frames worth of segments
            segment_count = max(1, len(data) // 25)
            segment_count = round(segment_count)
            segment_count = int(segment_count)

            segment_count = len(data) // 25
            segment_count = max(segment_count, 1)
            segment_count = min(segment_count, 4)
            segments = []

            start = 0
            if valid_regions:
                areas = [f.region.velocity for f in data]
                print("Using valid regions")
            else:
                # all regions are touching edge
                areas = [f.region.area for f in data]
            areas = np.array(areas)
            # weights = np.array(weights)
            area_sorted = np.argsort(areas)
            if len(areas) == 0:
                logging.error("NO weights for %s", track)
                return None, None, None

            index = 0

            indices = np.arange(len(data))
            # # indices = indices + max_start
            # segment_count = len(data) // 25
            # segment_count = max(segment_count, 1)
            # segment_count = min(segment_count, 4)
            segment_count = 1
            # print("Segments ", segment_count, len(weights))
            # logging.info(
            #     "Contours are %s median %s mean %s upper q %s",
            #     weights,
            #     np.median(weights),
            #     np.mean(weights),
            #     np.percentile(weights, 75),
            # )

            # weights = weights + areas
            # logging.info(
            #     "Contours are %s median %s mean %s upper q %s",
            #     weights,
            #     np.median(weights),
            #     np.mean(weights),
            #     np.percentile(weights, 75),
            # )
            # do something slightly smarter

            from ml_tools.preprocess import preprocess_frame, preprocess_movement

            preprocessed = []
            masses = []
            segment_frames = []
            # contours = weights.copy()

            # weights = weights[indices] ** 2

            # weights = np.float32(weights) / np.sum(weights)
            for seg in range(segment_count):
                frame_numbers = []
                mass = 0
                # choice = np.random.choice(
                #     indices,
                #     size=min(len(indices), 25),
                #     p=weights,
                #     replace=False,
                # )
                # choice = sorted_contours[-10:]
                # choice.sort()
                # segment_data = data[choice]
                # seg_weights = weights[choice]
                # print("COntour mean is ", np.mean(seg_weights))
                # segments.append(choice)
                choice = list(area_sorted[-25:])
                while len(choice) < 25:
                    missing = 25- len(choice)
                    choice.extend(area_sorted[-missing:])
                choice.sort()
                # choice = area_sorted[-25:]
                # segment_data= data[-25:]
                segment_data = data[choice]
                # sorted_contours = sorted_contours[:-25]
                preprocess_data = []
                for z, frame_data in enumerate(segment_data):
                    cropped_frame = preprocess_frame(
                        frame_data,
                        (self.params.frame_size, self.params.frame_size),
                        frame_data.region,
                        clip.background,
                        clip.crop_rectangle,
                        calculate_filtered=False,
                        filtered_norm_limits=filtered_norm_limits,
                        thermal_norm_limits=thermal_norm_limits,
                    )
                    # logging.info("adding %s contours %s",frame_data.frame_number,areas[z])
                    mass += frame_data.region.mass
                    preprocess_data.append(cropped_frame)
                    frame_numbers.append(frame_data.frame_number)
                frame_numbers.sort()
                segment_frames.append(frame_numbers)
                frames,samples = preprocess_movement(
                    preprocess_data,
                    self.params.square_width,
                    self.params.frame_size,
                    self.params.channels,
                    self.preprocess_fn,
                    sample=f"{clip.get_id()}-{track.get_id()}",
                )
                full_frames = []
                for sample in samples:
                    full_frames.append(frame_numbers[sample])
                logging.info("Used frame numbers %s ",full_frames)
                preprocessed.append(frames)
                if len(data) < 25:
                    mass = 25 * mass / len(data)
                masses.append(mass)
            preprocessed = np.array(preprocessed)
            logging.info("Using %s",frame_numbers)
            return [frame_numbers], preprocessed, masses

    def preprocess_segments(
        self,
        clip,
        track,
        max_segments=None,
        predict_from_last=None,
        segment_frames=None,
        dont_filter=False,
        min_segments=None,
    ):
        from ml_tools.preprocess import preprocess_frame, preprocess_movement

        track_data = {}
        segments = track.get_segments(
            self.params.square_width**2,
            ffc_frames=[] if dont_filter else clip.ffc_frames,
            repeats=1,
            segment_frames=segment_frames,
            segment_types=self.params.segment_types,
            from_last=predict_from_last,
            max_segments=max_segments,
            dont_filter=dont_filter,
            filter_by_fp=False,
            min_segments=min_segments,
        )
        frame_indices = set()
        for segment in segments:
            frame_indices.update(set(segment.frame_indices))
        frame_indices = list(frame_indices)
        frame_indices.sort()

        # should really be over whole track buts let just do the indices we predict of
        #  seems to make little different to just doing a min max normalization
        thermal_norm_limits = None
        filtered_norm_limits = None
        if self.params.diff_norm or self.params.thermal_diff_norm:
            min_diff = None
            max_diff = 0
            thermal_max_diff = None
            thermal_min_diff = None
            for i, region in enumerate(reversed(track.bounds_history)):
                if region.blank:
                    continue
                if region.width == 0 or region.height == 0:
                    logging.warn(
                        "No width or height for frame %s regoin %s",
                        region.frame_number,
                        region,
                    )
                    continue
                f = clip.get_frame(region.frame_number)
                if region.blank or region.width <= 0 or region.height <= 0 or f is None:
                    continue

                f.float_arrays()

                if self.params.thermal_diff_norm:
                    diff_frame = f.thermal - np.median(f.thermal)
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if thermal_min_diff is None or new_min < thermal_min_diff:
                        thermal_min_diff = new_min
                    if thermal_max_diff is None or new_max > thermal_max_diff:
                        thermal_max_diff = new_max
                if self.params.diff_norm:
                    diff_frame = region.subimage(f.filtered)
                    # - region.subimage(
                    #     clip.background
                    # )
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if min_diff is None or new_min < min_diff:
                        min_diff = new_min
                    if new_max > max_diff:
                        max_diff = new_max
            if self.params.thermal_diff_norm:
                thermal_norm_limits = (thermal_min_diff, thermal_max_diff)

            if self.params.diff_norm:
                filtered_norm_limits = (min_diff, max_diff)
        for frame_index in frame_indices:
            region = track.bounds_history[frame_index - track.start_frame]

            frame = clip.get_frame(region.frame_number)
            # filtered is calculated slightly different for tracking, set to null so preprocess can recalc it
            if frame is None:
                logging.error(
                    "Clasifying clip %s track %s can't get frame %s",
                    clip.get_id(),
                    track.get_id(),
                    region.frame_number,
                )
                raise Exception(
                    "Clasifying clip {} track {} can't get frame {}".format(
                        clip.get_id(), track.get_id(), region.frame_number
                    )
                )
            cropped_frame = preprocess_frame(
                frame,
                (self.params.frame_size, self.params.frame_size),
                region,
                clip.background,
                clip.crop_rectangle,
                calculate_filtered=False,
                filtered_norm_limits=filtered_norm_limits,
                thermal_norm_limits=thermal_norm_limits,
            )
            track_data[frame.frame_number] = cropped_frame
        features = None
        if self.params.mvm:
            from ml_tools.forestmodel import process_track as forest_process_track

            features = forest_process_track(
                clip, track, normalize=True, predict_from_last=predict_from_last
            )

        preprocessed = []
        masses = []
        for segment in segments:
            segment_frames = []
            for frame_i in segment.frame_indices:
                f = track_data[frame_i]
                # probably no need to copy
                segment_frames.append(f.copy())
            frames = preprocess_movement(
                segment_frames,
                self.params.square_width,
                self.params.frame_size,
                self.params.channels,
                self.preprocess_fn,
                sample=f"{clip.get_id()}-{track.get_id()}",
            )
            if frames is None:
                logging.warn("No frames to predict on")
                continue
            preprocessed.append(frames)
            masses.append(segment.mass)
        preprocessed = np.array(preprocessed)
        if self.params.mvm:
            features = features[np.newaxis, :]
            features = np.repeat(features, len(preprocessed), axis=0)
            preprocessed = [preprocessed, features]

        return [s.frame_indices for s in segments], preprocessed, masses


class NeuralInterpreter(Interpreter):
    TYPE = "Neural"

    def __init__(self, model_name):
        from openvino.inference_engine import IENetwork, IECore

        super().__init__(model_name)
        model_name = Path(model_name)
        # can use to test on PC
        # device = "CPU"
        device = "MYRIAD"
        model_xml = model_name.with_suffix(".xml")
        model_bin = model_name.with_suffix(".bin")
        ie = IECore()
        ie.set_config({}, device)
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.input_shape = net.input_info[self.input_blob].input_data.shape

        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)
        self.preprocess_fn = inc3_preprocess

    def predict(self, input_x):
        if input_x is None:
            return None
        input_x = np.float32(input_x)
        channels_last = input_x.shape[-1] == 3
        if channels_last:
            input_x = np.moveaxis(input_x, 3, 1)
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        return res

    def shape(self):
        return 1, self.input_shape


class LiteInterpreter(Interpreter):
    TYPE = "TFLite"

    def __init__(self, model_name):
        print("Loading lite")
        from ai_edge_litert.interpreter import Interpreter

        super().__init__(model_name)

        model_name = Path(model_name)
        model_name = model_name.with_suffix(".tflite")
        self.interpreter = Interpreter(str(model_name))

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.
        self.input = self.interpreter.get_input_details()[0]  # Model has single input.
        self.preprocess_fn = self.get_preprocess_fn()
        # inc3_preprocess

    def predict(self, input_x):
        input_x = np.float32(input_x)
        preds = []
        # only works on input of 1
        for data in input_x:
            self.interpreter.set_tensor(self.input["index"], data[np.newaxis, :])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output["index"])
            preds.append(pred)
        return pred

    def shape(self):
        return 1, self.input["shape"]


def inc3_preprocess(x):
    x /= 127.5
    x -= 1.0
    return x


def get_interpreter_from_path(model_file):
    logging.info("Loading %s", model_file)

    if model_file.suffix in [".keras", ".pb"]:
        from ml_tools.kerasmodel import KerasModel

        classifier = KerasModel()
        classifier.load_model(model_file)
    elif model_file.suffix == ".tflite":
        classifier = LiteInterpreter(model_file)
    elif model_file.suffix == ".pkl":
        from ml_tools.forestmodel import ForestModel

        classifier = ForestModel(model_file)
    return classifier


def get_interpreter(model):
    # model_name, type = os.path.splitext(model.model_file)

    logging.info(
        "Loading %s of type %s",
        model.model_file,
        model.type,
    )

    if model.type == LiteInterpreter.TYPE:
        classifier = LiteInterpreter(model.model_file)
    elif model.type == NeuralInterpreter.TYPE:
        classifier = NeuralInterpreter(model.model_file)
    elif model.type == "RandomForest":
        from ml_tools.forestmodel import ForestModel

        classifier = ForestModel(model.model_file)
    else:
        from ml_tools.kerasmodel import KerasModel

        classifier = KerasModel()
        classifier.load_model(model.model_file, weights=model.model_weights)
    classifier.id = model.id
    return classifier


c_i = 0


def get_contours(contour_image, frame_number):
    import cv2

    global c_i
    c_i += 1
    contour_image, stats = normalize(contour_image, new_max=255)
    cv2.imwrite(f"contours/og-{frame_number}.png", contour_image)

    image = cv2.GaussianBlur(np.uint8(contour_image), (15, 15), 0)

    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, image = cv2.threshold(image, 0, 255, flags)
    cv2.imwrite(f"contours/contours-{frame_number}.png", image)

    contours, _ = cv2.findContours(
        np.uint8(image),
        cv2.RETR_EXTERNAL,
        # cv2.CHAIN_APPROX_SIMPLE,
        cv2.CHAIN_APPROX_TC89_L1,
    )
    if len(contours) == 0:
        return 0

    contours = sorted(contours, key=lambda c: len(c), reverse=True)

    return len(contours[0])
