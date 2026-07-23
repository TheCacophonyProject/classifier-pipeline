import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, help="model file")

    parser.add_argument("model", type=Path, help="model file")
    parser.add_argument("source", type=Path, help="cptv file with txt")
    args = parser.parse_args()
    args.model = Path(args.model)
    args.source = Path(args.source)
    return args

# old model
# def build_model(metadata, weights_model):
#     base = tf.keras.applications.EfficientNetV2B3(
#         weights=None,
#         include_top=False,
#         input_tensor=tf.keras.Input((160, 160, 3)),
#     )
#     x = base.output
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     preds = tf.keras.layers.Dense(
#         len(metadata["labels"]), activation="softmax", name="prediction"
#     )(x)
#     model = tf.keras.models.Model(base.input, outputs=preds)
#     model.set_weights(weights_model.get_weights())
#     return model


def make_gradcam_heatmap(
    model, img_array, class_index, last_conv_layer_name="top_conv"
):
    """Returns (H, W) float32 heatmap in [0, 1]."""
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    grad_model.summary()
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)  # (1, h, w, filters)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)
    conv_outputs = conv_outputs[0]  # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    print("Heat map is ",heatmap.shape)
    return heatmap.numpy()


def overlay_heatmap_on_channel(channel, heatmap, alpha=0.4):
    """
    channel : (H, W) float array, any range
    heatmap  : (h, w) float [0,1]  — will be upsampled to match channel
    Returns  : (H, W, 3) uint8 image
    """
    H, W = channel.shape
    heatmap_up = tf.image.resize(heatmap[..., np.newaxis], (H, W)).numpy()[..., 0]
    # Normalise channel to [0, 1] for display
    cmin, cmax = channel.min(), channel.max()
    ch_norm = (channel - cmin) / (cmax - cmin + 1e-8)

    # Colour the heatmap (jet) and blend
    heat_rgb = cm.jet(heatmap_up)[..., :3]  # (H, W, 3)
    blended = (1 - alpha) * np.stack([ch_norm] * 3, axis=-1) + alpha * heat_rgb
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def preprocess_file(classifier, filename):
    from track.cliptrackextractor import ClipTrackExtractor, is_affected_by_ffc
    from track.clip import Clip
    from ml_tools.tools import load_clip_metadata, clear_session, CustomJSONEncoder
    from classify.trackprediction import Predictions

    from cptv_rs_python_bindings import CptvReader
    from piclassifier.motiondetector import RunningMean, SlidingWindow
    from piclassifier.cptvmotiondetector import CPTVMotionDetector
    from ml_tools.frame import Frame
    from ml_tools.preprocess import preprocess_frame_v2, preprocess_movement
    from datetime import datetime
    from config.trackingconfig import TrackingConfig
    from ml_tools.interpreter import get_frame_mask
    filename = Path(filename)
    meta_file = filename.with_suffix(".txt")
    has_metadata = meta_file.exists()
    if not filename.exists():
        logging.error("File %s not found.", filename)
        return False

    filename = Path(filename)
    if not has_metadata:
        logging.error("Must have metadata")
        raise Exception("No metadata found")

    # get segments here, or frames
    # only extra data for segments
    track_extractor = ClipTrackExtractor(
        {"thermal": TrackingConfig.get_type_defaults("thermal")},
        None,
        True,
        False,
        calculate_filtered=True,
    )

    clip = Clip(track_extractor.config, filename)
    meta_data = load_clip_metadata(meta_file)

    # rec_end = datetime.fromisoformat(meta_data["end_time"])

    clip.load_metadata(
        meta_data,
    )
    track_extractor.init_clip(clip)

    track_samples = {}
    track_data = {}

    for track in clip.tracks:
        print("Track is ",track)
        pred_frames = classifier.frames_for_prediction(clip, track)

        track_data[track.get_id()] = {
            "pred_frames": pred_frames,
            "limits": None,
            "frames": {},
            "track": track,
        }

        for seg in pred_frames:

            for r in seg.regions:
                frame_data = track_samples.setdefault(r.frame_number, {})
                frame_data[track.get_id()] = r
                # frame_samples.append(r)
    reader = CptvReader(str(clip.source_file))
    current_frame_num = 0
    running_mean = None
    thermal_window = SlidingWindow(CPTVMotionDetector.MEAN_FRAMES, "O")

    if classifier.params.thermal_diff_norm:
        logging.error("Thermal min diff is not implemented so will not be used")

    while True:
        frame = reader.next_frame()

        if frame is None:
            break
        if frame.background_frame:
            continue

        if current_frame_num in track_samples:
            thermal_median = np.median(frame.pix)
            filtered = frame.pix - track_extractor.background_alg.background

            f = Frame(frame.pix, filtered, current_frame_num)
            f.float_arrays()
            for track_id, region in track_samples[current_frame_num].items():
                f.region = region
                pre_f,_,_ = preprocess_frame_v2(
                    f,
                    classifier.params.frame_size,
                    region,
                    clip.crop_rectangle,
                    thermal_median,
                )
                pre_f.thermal *= 255
                pre_f.thermal_norm *= 255
                pre_f.filtered *= 255
                track_data[track_id]["frames"][region.frame_number] = pre_f

        # track_extractor.process_frame(clip, frame)
        is_ffc = is_affected_by_ffc(frame)
        oldest_thermal = thermal_window.oldest
        thermal_window.add(frame, is_ffc)

        if running_mean is None:
            running_mean = RunningMean([frame.pix], CPTVMotionDetector.MEAN_FRAMES)
        else:
            running_mean.add(frame.pix, oldest_thermal.pix)
        if not is_ffc:
            track_extractor.background_alg.process_frame(running_mean.mean())
        current_frame_num += 1
    i = 0
    for track_id, data in track_data.items():
        print("Track id is",track_id)
        i += 1
        pred_frames = data["pred_frames"]
        pred_frame_numbers = []
        preprocess_data = {"input_image": [], "input_mask": []}
        masses = []
        for segment in pred_frames:
            segment_frames = []
            for frame_i in segment.frame_indices:
                f = data["frames"][frame_i]
                segment_frames.append(f)

            frames = preprocess_movement(
                segment_frames,
                classifier.params.square_width,
                classifier.params.frame_size,
                classifier.params.channels,
                classifier.preprocess_fn,
                sample=f"{clip.get_id()}-{track_id}",
                enlarge=True
            )
            frame_mask = get_frame_mask(segment.frame_indices)
            # preprocess_data["input_image"].append(np.zeros_like(frames))
            preprocess_data["input_image"].append(frames)
            preprocess_data["input_mask"].append(frame_mask)

            masses.append(segment.mass)
            pred_frame_numbers.append(segment.frame_indices)
        if len(preprocess_data["input_image"]) == 0:
            logging.info("No prediction made for track %s", track_id)
            continue
            # dont think this should happen
        preprocess_data["input_image"] = np.array(preprocess_data["input_image"])
        preprocess_data["input_mask"] = np.array(preprocess_data["input_mask"])
        return preprocess_data


def init_logging():
    import sys

    fmt = "%(asctime)s %(process)d %(thread)s:%(levelname)7s %(message)s"

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    args = parse_args()
    init_logging()
    meta_f = args.model.with_suffix(".json")
    with meta_f.open("r") as f:
        metadata = json.load(f)

    from ml_tools.interpreter import get_interpreter_from_path

    classifier = get_interpreter_from_path(args.model)
    data = preprocess_file(classifier, args.source)
    labels = metadata["labels"]
    channel_names = ["Red", "Green", "Blue"]

    old_model = classifier.model

    if args.weights is not None:
        old_model.load_weights(args.weights)
    # model = build_model(metadata, old_model)
    old_model.summary()
    model = old_model
    # source = np.load(args.source)          # expected (H, W, 3) or (1, H, W, 3)
    # if source.ndim == 3:
    #     source = source[np.newaxis]        # → (1, H, W, 3)
    # img = source[0]                        # (H, W, 3) for display

    preds = model.predict(data)
    for pred, pred_image,pred_mask in zip(preds, data["input_image"],data["input_mask"]):
        print("Predictions:")
        for i, label in enumerate(labels):
            print(f"  {label}: {pred[i]*100:.1f}%")
        top_i = int(np.argmax(pred))
        top_label = labels[top_i]
        heatmap = make_gradcam_heatmap(
            model, {"input_image":np.expand_dims(pred_image,0),"input_mask":np.expand_dims(pred_mask,0)}, class_index=top_i,last_conv_layer_name="conv2d_1"
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        fig.suptitle(
            f"{top_label} ({pred[top_i]*100:.1f}%) — {args.source.name}", fontsize=12
        )

        for ci, ch_name in enumerate(channel_names):
            vis = overlay_heatmap_on_channel(pred_image[..., ci], heatmap)
            ax = axes[0][ci]
            ax.imshow(vis)
            ax.set_title(ch_name, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        out_path = args.source.parent / f"{args.source.stem}-gradcam.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
        break
        # plt.show()



if __name__ == "__main__":
    main()
