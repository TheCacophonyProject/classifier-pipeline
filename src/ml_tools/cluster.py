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
    args = parser.parse_args()
    return args

def main():
    args  =parse_args()
    init_logging()
    labels = ['bird', 'cat', 'deer', 'dog', 'false-positive', 'hedgehog', 'human', 'kiwi', 'leporidae', 'mustelid', 'penguin', 'possum', 'rodent', 'sheep', 'vehicle', 'wallaby','weka','chicken'] 
  
    #new_labels = extract_embeddings(args.dataset,args.model,args.output,weights = args.weights,included_labels = labels)
    label_f = args.output.with_name(f"{args.output.stem}-labels.npy")
    new_labels = np.load(label_f)
    logging.info("Loaded labels %s",new_labels)
    run_umap(args.model,args.output.with_name(f"{args.output.stem}-features.npy"),new_labels)
    return


def run_umap(model_file,features_file,labels):
    import numpy as np
    import pandas as pd
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import AgglomerativeClustering

    meta_file = model_file.with_suffix(".json")
    with open(meta_file) as f:
        meta = json.load(f)
    # labels = meta.get("labels", [])
    features = np.load(features_file)
    labels_file = features_file.with_name(f"{features_file.stem.replace("-features","-labels")}.npy")
    true_labels = np.load(labels_file)

    # dont do fps
    fp_index = np.where(labels == "false-positive")[0][0]
    labels = np.array(labels)

    item_mask = true_labels!=fp_index
    features = features[item_mask]
    true_labels = true_labels[item_mask]
    true_labels = labels[true_labels]

    logging.info("Features are %s labels %s",features.shape,true_labels.shape)
    
    # --- 2. Run Unsupervised UMAP ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)  # Notice: y is NOT passed here


    # calculate distance of groups and get a colour palette based of this
    centroids = []
    unique_labels = np.unique(true_labels)
    for label in unique_labels:
        mask = (true_labels == label)
        centroid = embedding[mask].mean(axis=0)
        centroids.append(centroid)
        
    centroids = np.array(centroids)
    num_neighborhoods = min(4, len(unique_labels)) 
    label_clustering = AgglomerativeClustering(n_clusters=num_neighborhoods)
    neighborhood_assignments = label_clustering.fit_predict(centroids)
    labels_df = pd.DataFrame({
        'label': unique_labels,
        'neighborhood': neighborhood_assignments
    }).sort_values(by='neighborhood').reset_index(drop=True)

    df = pd.DataFrame({
        'UMAP 1': embedding[:, 0],
        'UMAP 2': embedding[:, 1],
        'Class': true_labels ,

    })

    palette = list(sns.color_palette("tab20", n_colors=len(unique_labels)))
    labels_df['color'] = palette
    colour_mapping = dict(zip(labels_df['label'], labels_df['color']))

    # Setting a seed ensures your colors don't change every time you run the script
    # np.random.seed(42) 
    # np.random.shuffle(palette)
    plt.figure(figsize=(10, 8), dpi=300)
    sns.scatterplot(
        data=df, 
        x='UMAP 1', 
        y='UMAP 2', 
        hue='Class',          # Automatically maps text labels to unique colors
        palette=colour_mapping,
        style='Class',
        s=8,  
        alpha=1,        
        edgecolor='w'   
    )
    plt.title('EfficientNet Feature Space (Unsupervised UMAP)', fontsize=14)
    plt.savefig(features_file.with_suffix(".jpg"), dpi=300)

    import hdbscan


    # 2. Cluster with HDBSCAN
    # The lower the min_cluster_size and min_samples, the more granular the detection
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15, 
        min_samples=5, 
        gen_min_span_tree=True
    )
    clusterer.fit(embedding)

    # 3. Detect Anomalies
    # HDBSCAN assigns -1 to points that do not fall into any cluster
    anomaly_labels = clusterer.labels_
    anomalies = true_labels[anomaly_labels == -1]

    print(f"Detected {len(anomalies)} anomalies out of {len(true_labels)} data points.")
    print(anomalies)

def extract_embeddings(dataset_dir, model_file, output_file, weights = None,batch_size=32,included_labels = None):
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
    logging.info("Running on labels %s excluded %s remapped %s",labels,excluded_labels,remapped_labels)
    model = tf.keras.models.load_model(model_file)
    if weights is not None:
        logging.info("Loading weights %s",weights)
        model.load_weights(weights)
    model.trainable = False

    truncated = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    truncated.summary()

    input_shape = model.input.shape  # (batch, h, w, c)
    img_h = input_shape[1]
    num_channels = input_shape[-1]
    channels = [TrackChannels.thermal.name, TrackChannels.filtered.name, TrackChannels.filtered.name][:num_channels]

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
        channels=channels,
    )

    predictions = truncated.predict(dataset)
    true_labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in dataset])
    output_predictions = output_file.with_name(f"{output_file.stem}-features.npy")
    output_labels = output_file.with_name(f"{output_file.stem}-labels.npy")
    new_labels_out = output_file.with_name(f"{output_file.stem}-classes.npy")

    np.save(output_predictions, predictions)
    np.save(output_labels, true_labels)
    np.save(new_labels_out, new_labels)
    logging.info("New labls are %s",new_labels)
    logging.info(
        "Saved %d embeddings to %s and labels to %s",
        len(predictions),
        output_predictions,
        output_labels,
    )
    return new_labels
