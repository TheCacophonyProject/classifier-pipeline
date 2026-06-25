import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

from ml_tools import thermaldataset
from ml_tools.frame import TrackChannels
from ml_tools.tfdataset import get_dataset
import argparse
from ml_tools.logs import init_logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="Path to model file to use, will override config model",
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to model file to use, will override config model",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path to model file to use, will override config model",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        help="Path to models file to use, will override config model",
    )

    parser.add_argument(
        "--only-umap",
        action="store_true",
        help="Only run umap part",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_logging()
    labels = [
        "bird",
        "cat",
        "deer",
        "dog",
        "false-positive",
        "hedgehog",
        "human",
        "kiwi",
        "leporidae",
        "mustelid",
        "penguin",
        "possum",
        "rodent",
        "sheep",
        "vehicle",
        "wallaby",
        "weka",
        "chicken",
    ]
    if not args.only_umap:
        new_labels = extract_embeddings(
            args.dataset,
            args.model,
            args.output,
            weights=args.weights,
            included_labels=labels,
        )
    label_f = args.output.with_name(f"{args.output.stem}-classes.npy")
    new_labels = np.load(label_f)
    logging.info("Loaded labels %s", new_labels)
    filter_labels = ["false-positive"]
    run_umap(
        args.model,
        args.output.with_name(f"{args.output.stem}-features.npy"),
        new_labels,
        filter_labels,
    )
    return


def run_umap(model_file, features_file, labels, filter_labels=None):
    import numpy as np
    import pandas as pd
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import AgglomerativeClustering

    print("Filtering by ", filter_labels)

    meta_file = model_file.with_suffix(".json")
    with open(meta_file) as f:
        meta = json.load(f)
    # labels = meta.get("labels", [])
    features = np.load(features_file)
    labels_file = features_file.with_name(
        f"{features_file.stem.replace("-features","-labels")}.npy"
    )
    true_labels = np.load(labels_file)
    tracks_file = features_file.with_name(
        f"{features_file.stem.replace("-features","-tracks")}.npy"
    )
    tracks = np.load(tracks_file)
    filter_labels = []
    labels = np.array(labels)
    keep_indices = np.array([i for i, l in enumerate(labels) if l not in filter_labels])

    item_mask = np.isin(true_labels, keep_indices)
    features = features[item_mask]
    true_labels = true_labels[item_mask]
    true_labels = labels[true_labels]

    logging.info("Features are %s labels %s", features.shape, true_labels.shape)

    # detect anomalies
    reducer = umap.UMAP(n_neighbors=90, min_dist=0.0, n_components=6, metric="cosine")
    embedding = reducer.fit_transform(features)  # Notice: y is NOT passed here

    import hdbscan

    # 2. Cluster with HDBSCAN
    # The lower the min_cluster_size and min_samples, the more granular the detection
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=40, min_samples=3, gen_min_span_tree=True
    )
    clusterer.fit(embedding)

    # 3. Detect Anomalies
    # HDBSCAN assigns -1 to points that do not fall into any cluster
    anomaly_labels = clusterer.labels_
    anomalies = true_labels[anomaly_labels == -1]
    track_anomolies = tracks[anomaly_labels == -1]

    print(f"Detected {len(anomalies)} anomalies out of {len(true_labels)} data points.")
    # print(anomalies)
    anomalies_by_label = {}
    for label, track in zip(anomalies, track_anomolies):

        anomalies_by_label.setdefault(str(label), []).append(int(track))
    anomalies_file = features_file.with_name(
        f"{features_file.stem.replace('-features', '-anomalies')}.json"
    )
    with open(anomalies_file, "w") as f:
        json.dump(anomalies_by_label, f, indent=2)
    logging.info("Saved anomalies to %s", anomalies_file)

    # draw umap with 2 components
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    embedding = reducer.fit_transform(features)  # Notice: y is NOT passed here

    # calculate distance of groups and get a colour palette based of this
    centroids = []
    unique_labels = np.unique(true_labels)
    for label in unique_labels:
        mask = true_labels == label
        centroid = embedding[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    num_neighborhoods = min(4, len(unique_labels))
    label_clustering = AgglomerativeClustering(n_clusters=num_neighborhoods)
    neighborhood_assignments = label_clustering.fit_predict(centroids)
    labels_df = (
        pd.DataFrame({"label": unique_labels, "neighborhood": neighborhood_assignments})
        .sort_values(by="neighborhood")
        .reset_index(drop=True)
    )

    df = pd.DataFrame(
        {
            "UMAP 1": embedding[:, 0],
            "UMAP 2": embedding[:, 1],
            "Class": true_labels,
            "Color": np.where(anomaly_labels == -1, "anomaly", true_labels),
        }
    )

    palette = list(sns.color_palette("tab20", n_colors=len(unique_labels)))
    labels_df["color"] = palette
    colour_mapping = dict(zip(labels_df["label"], labels_df["color"]))
    colour_mapping["anomaly"] = "black"

    # Setting a seed ensures your colors don't change every time you run the script
    # np.random.seed(42)
    # np.random.shuffle(palette)
    n_labels = len(unique_labels)
    fig_height = max(8, n_labels * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=600)
    sns.scatterplot(
        data=df,
        x="UMAP 1",
        y="UMAP 2",
        hue="Color",
        palette=colour_mapping,
        style="Class",
        s=8,
        alpha=1,
        edgecolor="w",
        ax=ax,
    )
    ax.set_title("EfficientNet Feature Space (Unsupervised UMAP)", fontsize=14)
    legend = ax.get_legend()
    legend.set_bbox_to_anchor((0, 0.5))
    legend.set_loc("center right")
    ax.legend(
        handles=legend.legend_handles,
        labels=[t.get_text() for t in legend.get_texts()],
        loc="center right",
        bbox_to_anchor=(0, 0.5),
        title="Class",
        framealpha=0.8,
    )
    fig.savefig(features_file.with_suffix(".jpg"), dpi=600, bbox_inches="tight")


def extract_embeddings(
    dataset_dir,
    model_file,
    output_file,
    weights=None,
    batch_size=32,
    included_labels=None,
):
    """
    Load a keras model, remove its last 2 layers, run a thermal dataset through the
    truncated model, and save the resulting embeddings and true class indices to numpy files.

    Args:
        dataset_dir: directory containing .tfrecord files
        model_file: path to a saved keras model (.keras or SavedModel directory)
        output_predictions: path for the embeddings numpy file (.npy)
        output_labels: path for the true label indices numpy file (.npy)
        batch_size: inference batch size
    Returns:
        list of label names corresponding to the saved label indices
    """
    dataset_dir = Path(dataset_dir)
    model_file = Path(model_file)

    meta_file = model_file.with_suffix(".json")
    # if not meta_file.exists():
    # meta_file = dataset_dir / "training-meta.json"
    with open(meta_file) as f:
        meta = json.load(f)

    trianing_meta_f = dataset_dir.parent / "training-meta.json"
    with trianing_meta_f.open("r") as f:
        training_meta = json.load(f)
    labels = training_meta.get("labels", [])
    excluded_labels = meta.get("excluded_labels") or []
    remapped_labels = meta.get("remapped_labels") or {}
    for l in included_labels:
        if l in excluded_labels:
            excluded_labels.remove(l)
        if l in remapped_labels:
            del remapped_labels[l]
    logging.info(
        "Running on labels %s excluded %s remapped %s",
        labels,
        excluded_labels,
        remapped_labels,
    )
    model = tf.keras.models.load_model(model_file)
    if weights is not None:
        logging.info("Loading weights %s", weights)
        model.load_weights(weights)
    model.trainable = False

    truncated = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    truncated.summary()

    input_shape = model.input.shape  # (batch, h, w, c)
    img_h = input_shape[1]
    num_channels = input_shape[-1]
    # channels = [TrackChannels.thermal.name, TrackChannels.filtered.name, TrackChannels.filtered.name][:num_channels]

    dataset, _, new_labels, _ = get_dataset(
        thermaldataset.load_dataset,
        dataset_dir,
        labels,
        batch_size=batch_size,
        image_size=(img_h, img_h),
        shuffle=False,
        excluded_labels=excluded_labels,
        remapped_labels=remapped_labels,
        deterministic=True,
        include_track=True,
        # channels=channels,
    )

    predictions = truncated.predict(dataset)
    true_labels = []
    bird_index = labels.index("bird")
    tracks = []
    for _, batch_y in dataset:
        track_batch = batch_y[1]
        tracks.extend(track_batch)
        label_batch = batch_y[0]
        for y in label_batch:
            non_zero = np.nonzero(y.numpy())[0]
            if len(non_zero) > 1:
                print("non zero is ", non_zero)
                # at the moment only bird or something else as multi
                y = non_zero[non_zero != bird_index][0]
                print("FOr non zero", non_zero, " Using ", new_labels[y])
            else:
                y = non_zero[0]
            true_labels.append(y)
    tracks = np.array(tracks)
    true_labels = np.array(true_labels)
    output_predictions = output_file.with_name(f"{output_file.stem}-features.npy")
    output_labels = output_file.with_name(f"{output_file.stem}-labels.npy")
    new_labels_out = output_file.with_name(f"{output_file.stem}-classes.npy")
    tracks_out = output_file.with_name(f"{output_file.stem}-tracks.npy")

    np.save(output_predictions, predictions)
    np.save(output_labels, true_labels)
    np.save(new_labels_out, new_labels)
    np.save(tracks_out, tracks)

    logging.info("New labls are %s", new_labels)
    logging.info(
        "Saved %d embeddings to %s and labels to %s",
        len(predictions),
        output_predictions,
        output_labels,
    )
    return new_labels
