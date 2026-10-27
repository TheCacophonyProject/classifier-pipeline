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
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Generate a gradcam image for every label, not just the top prediction",
    )
    parser.add_argument(
        "--occlusion",
        action="store_true",
        help="Generate a per tile occlusion map instead of gradcam",
    )

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
    model, img_array, class_index, last_conv_layer_name="efficientnetv2-b3"
):
    """Returns (H, W) float32 heatmap in [0, 1]."""
    conv_layer = model.get_layer(last_conv_layer_name)

    # A single combined Model spanning a nested sub-model (efficientnetv2-b3)
    # loses the gradient link to that sub-model's .output tensor under Keras 3,
    # so tape.gradient() below comes back None. Splitting the graph into a
    # pre-model (up to conv_layer) and a post-model (everything after,
    # reapplied to an explicitly watched tensor) keeps the link intact.
    pre_model = tf.keras.models.Model(model.inputs, conv_layer.output)

    layer_index = model.layers.index(conv_layer)
    remaining_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
    y = remaining_input
    for layer in model.layers[layer_index + 1 :]:
        y = layer(y, training=False)
    post_model = tf.keras.models.Model(remaining_input, y)

    conv_outputs = pre_model(img_array, training=False)
    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        predictions = post_model(conv_outputs, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)  # (1, h, w, filters)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)
    conv_outputs = conv_outputs[0]  # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    print("Heat map is ", heatmap.shape)
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


def print_channel_contributions(model, img_array, class_index, label, channel_names):
    """Prints how much each input channel drives the class score, via
    Grad x Input summed over space — a cheap per-channel importance measure."""
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model({"input_image": img_tensor}, training=False)
        score = predictions[:, class_index]

    grads = tape.gradient(score, img_tensor)  # (1, H, W, C)
    contributions = tf.reduce_sum(tf.abs(grads * img_tensor), axis=(0, 1, 2)).numpy()
    total = contributions.sum() + 1e-8

    print(f"Channel contribution to '{label}' score:")
    for name, contrib in zip(channel_names, contributions):
        print(f"  {name}: {contrib:.4f} ({100 * contrib / total:.1f}%)")


def save_gradcam_for_label(
    model,
    pred_image,
    label_index,
    label,
    pred_score,
    out_path,
    channel_names,
):
    print_channel_contributions(
        model,
        np.expand_dims(pred_image, 0),
        label_index,
        label,
        channel_names,
    )
    heatmap = make_gradcam_heatmap(
        model,
        {
            "input_image": np.expand_dims(pred_image, 0),
        },
        class_index=label_index,
        last_conv_layer_name="efficientnetv2-b3",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    fig.suptitle(f"{label} ({pred_score*100:.1f}%)", fontsize=12)

    for ci, ch_name in enumerate(channel_names):
        vis = overlay_heatmap_on_channel(pred_image[..., ci], heatmap)
        ax = axes[0][ci]
        ax.imshow(vis)
        ax.set_title(ch_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def make_occlusion_map(model, img, square_width, pad_with=0):
    """Replaces each tile with padding and records how much every class score drops.
    Tiles that are already all padding are skipped and left with a drop of 0.
    Returns (base scores (labels,), drops (square_width, square_width, labels))."""
    tile = img.shape[0] // square_width
    batch = [img]
    tile_positions = []
    for r in range(square_width):
        for c in range(square_width):
            tile_slice = (
                slice(r * tile, (r + 1) * tile),
                slice(c * tile, (c + 1) * tile),
            )
            if np.all(img[tile_slice] == pad_with):
                continue
            occluded = img.copy()
            occluded[tile_slice] = pad_with
            batch.append(occluded)
            tile_positions.append((r, c))
    scores = model.predict({"input_image": np.array(batch)}, verbose=0)
    base = scores[0]
    drops = np.zeros((square_width, square_width, scores.shape[1]), dtype=np.float32)
    for (r, c), occluded_scores in zip(tile_positions, scores[1:]):
        drops[r, c] = base - occluded_scores
    return base, drops


def save_occlusion_for_label(
    pred_image,
    base,
    drops,
    label,
    pred_score,
    out_path,
    display_channel,
    display_name,
    square_width,
):
    """Whole tiles are occluded across all channels, so there is one map to draw.
    It is shown over display_channel only."""
    np.set_printoptions(precision=3, suppress=True)
    print(f"Occlusion score drops for '{label}' (base score {base:.3f}):")
    print(drops)

    # only tiles that support the label are coloured, scaled to the biggest drop
    heatmap = np.maximum(drops, 0) / (drops.max() + 1e-8)
    heatmap = np.float32(heatmap)
    tile = pred_image.shape[0] // square_width

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(
        f"{label} ({pred_score*100:.1f}%) occlusion, base score {base:.2f}",
        fontsize=12,
    )
    vis = overlay_heatmap_on_channel(
        pred_image[..., display_channel],
        np.kron(heatmap, np.ones((tile, tile), dtype=np.float32)),
    )
    ax.imshow(vis)
    for r in range(square_width):
        for c in range(square_width):
            ax.text(
                (c + 0.5) * tile,
                (r + 0.5) * tile,
                f"{drops[r, c]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    ax.set_title(display_name, fontsize=9)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def save_raw_channels(pred_image, channel_names, out_path):
    """Saves each input channel as its own greyscale panel, with no overlay."""
    fig, axes = plt.subplots(
        1, len(channel_names), figsize=(4 * len(channel_names), 4), squeeze=False
    )
    for ci, ch_name in enumerate(channel_names):
        channel = pred_image[..., ci]
        ax = axes[0][ci]
        ax.imshow(channel, cmap="gray")
        ax.set_title(
            f"{ch_name} ({channel.min():.0f} - {channel.max():.0f})", fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def preprocess_file(classifier, filename):
    from track.cliptrackextractor import ClipTrackExtractor
    from piclassifier.cptvmotiondetector import is_affected_by_ffc

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
        TrackingConfig.get_defaults(),
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
        print("Track is ", track)
        pred_frames = classifier.frames_for_prediction(clip, track)

        track_data[track.id] = {
            "pred_frames": pred_frames,
            "limits": None,
            "frames": {},
            "track": track,
        }

        for seg in pred_frames:

            for r in seg.regions:
                frame_data = track_samples.setdefault(r.frame_number, {})
                frame_data[track.id] = r
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
            filtered = np.float32(frame.pix) - track_extractor.background_alg.background
            f = Frame(frame.pix, filtered, current_frame_num)
            f.float_arrays()
            for track_id, region in track_samples[current_frame_num].items():
                f.region = region
                pre_f, _, _ = preprocess_frame_v2(
                    f,
                    classifier.params.frame_size,
                    region,
                    clip.crop_rectangle,
                    enlarge=True,
                    new_max=255.0,
                    thermal_norm=classifier.params.thermal_norm,
                )
                # pre_f.thermal *= 255
                # pre_f.thermal_norm *= 255
                # pre_f.filtered *= 255
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
        print("Track id is", track_id)

        i += 1
        i += 1
        pred_frames = data["pred_frames"]
        pred_frame_numbers = []
        preprocess_data = {"input_image": []}
        masses = []
        for segment in pred_frames:
            segment_frames = []
            for frame_i in segment.frame_indices:
                f = data["frames"][frame_i]
                segment_frames.append(f)

            frames = preprocess_movement(
                segment_frames,
                classifier.params.square_width,
                classifier.params.frame_size * 2,
                classifier.params.channels,
                classifier.preprocess_fn,
                sample=f"{clip.id}-{track_id}",
                pad_with=0,
            )
            # preprocess_data["input_image"].append(np.zeros_like(frames))
            preprocess_data["input_image"].append(frames)

            masses.append(segment.mass)
            pred_frame_numbers.append(segment.frame_indices)
        if len(preprocess_data["input_image"]) == 0:
            logging.info("No prediction made for track %s", track_id)
            continue
            # dont think this should happen
        preprocess_data["input_image"] = np.array(preprocess_data["input_image"])
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

    model = classifier.model

    if args.weights is not None:
        print("Loading ", args.weights)
        model.load_weights(args.weights)
    # model = build_model(metadata, old_model)
    model.summary()

    from modelevaluate import has_activation, add_sigmoid_output

    # occlusion is measured on logits where possible, so confident predictions
    # aren't flattened by a saturated sigmoid
    score_model = model
    if not has_activation(model):
        print("Added sigmoid output")
        model = add_sigmoid_output(model)
        model.summary()

    # source = np.load(args.source)          # expected (H, W, 3) or (1, H, W, 3)
    # if source.ndim == 3:
    #     source = source[np.newaxis]        # → (1, H, W, 3)
    # img = source[0]                        # (H, W, 3) for display

    preds = model.predict(data)
    for pred, pred_image in zip(preds, data["input_image"]):
        print(pred_image.shape)
        print("Predictions:")
        for i, label in enumerate(labels):
            print(f"  {label}: {pred[i]*100:.1f}%")
        top_i = int(np.argmax(pred))
        save_raw_channels(
            pred_image,
            [str(ch) for ch in classifier.params.channels],
            args.source.parent / f"{args.source.stem}-raw.png",
        )

        if args.occlusion:
            suffix = "occlusion"
            # the animal shows most clearly in the filtered channel
            model_channels = [str(ch) for ch in classifier.params.channels]
            display_channel = (
                model_channels.index("filtered")
                if "filtered" in model_channels
                else len(model_channels) - 1
            )
            display_name = model_channels[display_channel]
            # one set of predictions gives the drops for every label
            base, drops = make_occlusion_map(
                score_model, pred_image, classifier.params.square_width
            )

            def save_fn(label_i, out_path):
                save_occlusion_for_label(
                    pred_image,
                    base[label_i],
                    drops[..., label_i],
                    labels[label_i],
                    pred[label_i],
                    out_path,
                    display_channel,
                    display_name,
                    classifier.params.square_width,
                )

        else:
            suffix = "gradcam"

            def save_fn(label_i, out_path):
                save_gradcam_for_label(
                    model,
                    pred_image,
                    label_i,
                    labels[label_i],
                    pred[label_i],
                    out_path,
                    channel_names,
                )

        if args.all_labels:
            for label_i, label in enumerate(labels):
                save_fn(
                    label_i,
                    args.source.parent / f"{args.source.stem}-{suffix}-{label}.png",
                )
        else:
            save_fn(top_i, args.source.parent / f"{args.source.stem}-{suffix}.png")
        break
        # plt.show()


if __name__ == "__main__":
    main()
