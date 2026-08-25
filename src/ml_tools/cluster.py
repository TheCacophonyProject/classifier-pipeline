import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

from ml_tools import thermaldataset
from ml_tools.frame import TrackChannels
from ml_tools.tfdataset import get_dataset, apply_label_mapping

import argparse
from ml_tools.logs import init_logging
import hdbscan


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

    parser.add_argument(
        "--train-isoforest",
        action="store_true",
        help="Train an IsolationForest novelty detector on the extracted features",
    )
    parser.add_argument(
        "--hdbs-only",
        action="store_true",
        help="Re run hdbs on umap",
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
    if not args.only_umap and not args.hdbs_only:
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

    if args.train_isoforest:
        train_isolation_forest(
            args.output.with_name(f"{args.output.stem}-features.npy")
        )
        return

    # print("USING HDB TESTING")
    if args.hdbs_only:
        hdbs_load(
            args.output.with_name(f"{args.output.stem}-features.npy"),
            new_labels,
            filter_labels,
        )
        return
    run_umap(
        args.model,
        args.output.with_name(f"{args.output.stem}-features.npy"),
        new_labels,
        filter_labels,
    )
    return


def train_isolation_forest(features_file):
    from sklearn.ensemble import IsolationForest
    import joblib

    features = np.load(features_file)
    logging.info("Training IsolationForest on features %s", features.shape)

    iso_forest = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42
    )
    iso_forest.fit(features)

    iso_file = features_file.with_name(features_file.stem + "-isoforest.pkl")
    joblib.dump(iso_forest, iso_file)
    logging.info("Saved IsolationForest to %s", iso_file)
    return iso_forest


def hdbs_load(features_file, labels, filter_labels=None):

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

    embedding_file = features_file.with_name(
        f"{features_file.stem.replace('-features', '-umap2d')}.npy"
    )
    embedding = np.load(embedding_file)
    import joblib

    hdb_file = features_file.with_name(features_file.stem + "-hdb.pkl")
    # clusterer = joblib.load(hdb_file)
    # 2. Cluster with HDBSCAN
    # The lower the min_cluster_size and min_samples, the more granular the detection
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=3,
        gen_min_span_tree=True,
    )
    clusterer.fit(embedding)
    joblib.dump(clusterer, hdb_file)

    find_mislabeled_points(clusterer, true_labels, tracks, features_file)


def run_umap(model_file, features_file, labels, filter_labels=None):
    import numpy as np
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
    reducer = umap.UMAP(n_neighbors=45, min_dist=0.0, n_components=6, metric="cosine")
    embedding = reducer.fit_transform(features)  # Notice: y is NOT passed here

    embedding_file = features_file.with_name(
        f"{features_file.stem.replace('-features', '-umap2d')}.npy"
    )
    np.save(embedding_file, embedding)
    logging.info("Saved 2D UMAP embedding to %s", embedding_file)

    import hdbscan

    # 2. Cluster with HDBSCAN
    # The lower the min_cluster_size and min_samples, the more granular the detection
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=3,
        gen_min_span_tree=True,
    )
    clusterer.fit(embedding)
    import joblib

    hdb_file = features_file.with_name(features_file.stem + "-hdb.pkl")

    joblib.dump(clusterer, hdb_file)

    find_mislabeled_points(clusterer, true_labels, tracks, features_file)
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
    reducer = umap.UMAP(n_neighbors=45, min_dist=0.1, n_components=2)
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


# These labels are treated as one group when working out whether a majority
# label dominates a cluster, since bird/chicken/penguin can be visually similar.
BIRD_FAMILY_LABELS = {"bird", "chicken", "penguin","weka"}


def find_mislabeled_points(clusterer, y_true, tracks, features_file):
    probabilities = clusterer.probabilities_
    hdbscan_labels = clusterer.labels_
    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    print(f"HDBSCAN found {n_clusters} clusters")
    noise_pct = (hdbscan_labels == -1).mean() * 100
    print(f"Noise points: {noise_pct:.1f}%")
    # Create a summary dataframe for easier inspection
    track_ids = tracks[:, 0]
    source_ids = tracks[:, 1]

    results = pd.DataFrame(
        {
            "index": range(len(y_true)),
            "y_true": y_true,
            "tracks": tracks,
            "source_ids": source_ids,

            "hdbscan_cluster": hdbscan_labels,
            "confidence": probabilities,
        }
    )
    # Flag 1: The point was rejected entirely by HDBSCAN and marked as noise (-1)
    noise_flags = results[results["hdbscan_cluster"] == -1]

    # Flag 2: The point sits firmly inside an HDBSCAN cluster,
    # but its group features disagree with its y_true label.
    # We filter for high confidence (> 0.8) to ensure it's deep inside that foreign cluster.
    mismatch_flags = []
    cluster_to_class_map = {-1: "Anomaly"}
    ambiguous_clusters = []
    # Total occurrences of each label across the whole dataset, used to work out
    # what share of a label's points ended up in any given cluster.
    total_label_counts = results["y_true"].value_counts()
    # Tally, per label, what percentage of its points landed in each cluster.
    label_cluster_tally = {label: {} for label in total_label_counts.index}
    # Tally, per cluster, what percentage of that cluster each label makes up.
    cluster_label_tally = {}
    for cluster_id in results["hdbscan_cluster"].unique():
        if cluster_id == -1:
            continue

        # Isolate the current cluster
        cluster_subset = results[results["hdbscan_cluster"] == cluster_id]

        # Calculate the percentage distribution of labels inside this cluster
        label_counts = cluster_subset["y_true"].value_counts(normalize=True)
        majority_label = label_counts.index[0]
        majority_percentage = label_counts.iloc[0]
        cluster_label_tally[cluster_id] = label_counts.to_dict()

        # bird/chicken/penguin can be visually similar, so let them combine to
        # form a majority even when no single one of them dominates alone.
        bird_family_percentage = label_counts[
            label_counts.index.isin(BIRD_FAMILY_LABELS)
        ].sum()
        is_bird_family_majority = bird_family_percentage > majority_percentage
        if is_bird_family_majority:
            majority_label = "bird/chicken/penguin"
            majority_percentage = bird_family_percentage

        # If the top label doesn't dominate (e.g., less than 70% of the cluster), flag it
        other_labels = label_counts.iloc[1:]

        ambiguous_clusters.append(cluster_id)

        # For each label present in this cluster, what fraction of that label's
        # total occurrences in the whole dataset landed here.
        label_cluster_counts = cluster_subset["y_true"].value_counts()
        label_coverage = (
            label_cluster_counts / total_label_counts.reindex(label_cluster_counts.index)
        ).sort_values(ascending=False)
        for label, pct in label_coverage.items():
            label_cluster_tally[label][cluster_id] = pct
        label_breakdown = ", ".join(
            f"{label}:{pct:.1%} of label, {label_counts[label]:.1%} of cluster"
            for label, pct in sorted(
                label_coverage.items(), key=lambda item: label_counts[item[0]], reverse=True
            )
            if label_counts[label] > 0.20 or pct > 0.50
        )

        
        # A soft majority isn't actually ambiguous if this cluster is where most of
        # that label's data lives; only flag it when neither signal is strong.
        if is_bird_family_majority:
            majority_coverage = label_coverage[
                label_coverage.index.isin(BIRD_FAMILY_LABELS)
            ].sum()
        else:
            majority_coverage = label_coverage[majority_label]
        if majority_percentage < 0.70 and majority_coverage < 0.50:
            print(
                f"⚠️ Cluster {cluster_id} is ambiguous! Top label '{majority_label}' is only "
                f"{majority_percentage:.1%} of the cluster and {majority_coverage:.1%} of its own data"
            )
            for label, percentage in other_labels[other_labels > 0.20].items():
                print(f"    Other label '{label}' is {percentage:.1%}")
            cluster_to_class_map[cluster_id] = f"Ambiguous: {label_breakdown}"
        else:
            cluster_to_class_map[cluster_id] = label_breakdown

        # Find the dominant (majority) true label in this density cluster. When
        # bird/chicken/penguin combine to form the majority, any of the three counts.
        if is_bird_family_majority:
            is_majority_label = cluster_subset["y_true"].isin(BIRD_FAMILY_LABELS)
        else:
            majority_true_label = cluster_subset["y_true"].mode()[0]
            is_majority_label = cluster_subset["y_true"] == majority_true_label

        # A label whose data mostly lives in this cluster isn't really foreign here
        # either, even if it isn't the (combined) majority label.
        is_majority_label = is_majority_label | (
            cluster_subset["y_true"].map(label_coverage) > 0.50
        )

        # Flag points in this cluster that contradict the majority label with high certainty
        intruders = cluster_subset[
            (~is_majority_label) & (cluster_subset["confidence"] > 0.8)
        ]
        mismatch_flags.append(intruders)

    # Also tally noise (cluster -1) so the per-label breakdown accounts for all points.
    noise_subset = results[results["hdbscan_cluster"] == -1]
    noise_label_counts = noise_subset["y_true"].value_counts()
    noise_coverage = noise_label_counts / total_label_counts.reindex(noise_label_counts.index)
    for label, pct in noise_coverage.items():
        label_cluster_tally[label][-1] = pct
    cluster_label_tally[-1] = noise_subset["y_true"].value_counts(normalize=True).to_dict()

    print("\nPer-label cluster distribution:")
    for label in total_label_counts.index:
        breakdown = ", ".join(
            f"cluster {cluster_id}: {pct:.1%}"
            for cluster_id, pct in sorted(
                label_cluster_tally[label].items(), key=lambda item: item[1], reverse=True
            )
        )
        print(f"  {label}: {breakdown}")

    print("\nPer-cluster label composition:")
    for cluster_id in sorted(cluster_label_tally):
        breakdown = ", ".join(
            f"{label}: {pct:.1%}"
            for label, pct in sorted(
                cluster_label_tally[cluster_id].items(), key=lambda item: item[1], reverse=True
            )
        )
        print(f"  Cluster {cluster_id}: {breakdown}")

    mismatch_df = pd.concat(mismatch_flags)
    mismatch_df["mapped_to"] = [
        cluster_to_class_map[cluster_id]
        for cluster_id in mismatch_df["hdbscan_cluster"]
    ]

    # Group occurrences of the same track together, collecting their
    # confidences and mapped_to labels into arrays.
    grouped_df = (
        mismatch_df.groupby("tracks")
        .agg(
            y_true=("y_true", "first"),
            source_ids=("source_ids", list),
            hdbscan_cluster=("hdbscan_cluster", list),
            confidence=("confidence", list),
            mapped_to=("mapped_to", list),
        )
        .reset_index()
    )

    def best_confidence_per_label(mapped_to, confidence):
        # Keep only the highest-confidence occurrence of each distinct label,
        # sorted alphabetically by label so mapped_to and confidence stay aligned
        best = {}
        for label, conf in zip(mapped_to, confidence):
            if label not in best or conf > best[label]:
                best[label] = conf
        sorted_items = sorted(best.items(), key=lambda item: item[0])
        labels = [label for label, _ in sorted_items]
        confidences = [conf for _, conf in sorted_items]
        return pd.Series({"mapped_to": labels, "confidence": confidences})

    grouped_df[["mapped_to", "confidence"]] = grouped_df.apply(
        lambda row: best_confidence_per_label(row["mapped_to"], row["confidence"]),
        axis=1,
    )

    grouped_df = grouped_df.sort_values(by="mapped_to", key=lambda col: col.map(tuple))

    mismatch_file = features_file.with_name(features_file.stem + "-mismatch.csv")

    print("saving mismatches to ", mismatch_file)
    grouped_df.to_csv(mismatch_file, index=False)

    cluster_map_df = pd.DataFrame(
        sorted(cluster_to_class_map.items()),
        columns=["cluster_id", "mapped_class"],
    )
    with open(mismatch_file, "a") as f:
        f.write("\ncluster_to_class_map\n")
        cluster_map_df.to_csv(f, index=False)
        f.write("\nPer-label cluster distribution\n")
        for label in total_label_counts.index:
            breakdown = ", ".join(
                f"cluster {cluster_id}: {pct:.1%}"
                for cluster_id, pct in sorted(
                    label_cluster_tally[label].items(), key=lambda item: item[1], reverse=True
                )
            )
            f.write(f"{label}: {breakdown}\n")
        f.write("\nPer-cluster label composition\n")
        for cluster_id in sorted(cluster_label_tally):
            breakdown = ", ".join(
                f"{label}: {pct:.1%}"
                for label, pct in sorted(
                    cluster_label_tally[cluster_id].items(), key=lambda item: item[1], reverse=True
                )
            )
            f.write(f"Cluster {cluster_id}: {breakdown}\n")
    # Tracks whose mismatches were never explained away by an ambiguous
    # cluster are the ones most likely to be genuinely mislabeled.
    no_ambiguous_df = grouped_df[
        grouped_df["mapped_to"].apply(
            lambda labels: not any("Ambiguous" in label for label in labels)
        )
    ]
    mismatched_tracks_file = features_file.with_name("mismatched-tracks.txt")
    print("saving mismatched tracks to ", mismatched_tracks_file)
    with open(mismatched_tracks_file, "w") as f:
        for track,source in zip(no_ambiguous_df["tracks"],no_ambiguous_df["source_ids"]):
            f.write(f"{track}-{source}\n")
        f.write("\ncluster_to_class_map\n")
        for cluster_id, mapped_class in sorted(cluster_to_class_map.items()):
            f.write(f"{cluster_id}: {mapped_class}\n")
        f.write("\nPer-label cluster distribution\n")
        for label in total_label_counts.index:
            breakdown = ", ".join(
                f"cluster {cluster_id}: {pct:.1%}"
                for cluster_id, pct in sorted(
                    label_cluster_tally[label].items(), key=lambda item: item[1], reverse=True
                )
            )
            f.write(f"{label}: {breakdown}\n")
        f.write("\nPer-cluster label composition\n")
        for cluster_id in sorted(cluster_label_tally):
            breakdown = ", ".join(
                f"{label}: {pct:.1%}"
                for label, pct in sorted(
                    cluster_label_tally[cluster_id].items(), key=lambda item: item[1], reverse=True
                )
            )
            f.write(f"Cluster {cluster_id}: {breakdown}\n")


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

    trianing_meta_f = dataset_dir / "training-meta.json"
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
    model = tf.keras.models.load_model(model_file, compile=False)
    if weights is not None:
        logging.info("Loading weights %s", weights)
        model.load_weights(weights)
    model.trainable = False

    truncated = tf.keras.models.Model(
        model.inputs,
        model.get_layer("global_average_pooling2d").output,
    )
    truncated.summary()
    print(model.inputs)
    input_shape = model.inputs[0].shape  # (batch, h, w, c)
    img_h = input_shape[1] // 2
    num_channels = input_shape[-1]
    # channels = [TrackChannels.thermal.name, TrackChannels.filtered.name, TrackChannels.filtered.name][:num_channels]
    labels, tf_mappings = apply_label_mapping(labels, excluded_labels, remapped_labels)
    new_labels = labels
    dataset, epoch_size = get_dataset(
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
        tf_mappings=tf_mappings,
        # channels=channels,
    )

    predictions = truncated.predict(dataset)
    true_labels = []
    bird_index = labels.index("bird")
    tracks = []
    source_ids = []
    for _, batch_y in dataset:
        track_batch = batch_y[1]
        source_batch = batch_y[2]
        sources = [int(b.decode('utf-8')) for b in source_batch.numpy()]

        tracks.extend(track_batch)
        source_ids.extend(sources)

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
    source_ids = np.array(source_ids)
    true_labels = np.array(true_labels)
    output_predictions = output_file.with_name(f"{output_file.stem}-features.npy")
    output_labels = output_file.with_name(f"{output_file.stem}-labels.npy")
    new_labels_out = output_file.with_name(f"{output_file.stem}-classes.npy")
    tracks_out = output_file.with_name(f"{output_file.stem}-tracks.npy")

    np.save(output_predictions, predictions)
    np.save(output_labels, true_labels)
    np.save(new_labels_out, new_labels)
    np.save(tracks_out, np.stack((tracks,source_ids),axis=1))

    logging.info("New labls are %s", new_labels)
    logging.info(
        "Saved %d embeddings to %s and labels to %s",
        len(predictions),
        output_predictions,
        output_labels,
    )
    return new_labels
