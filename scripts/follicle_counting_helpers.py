import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from pathlib import Path
import os


def detect_follicles(
        adata,
        eps=55,
        min_samples=20,
        min_cluster_size=50,
):
    mask = (
            (adata.obs["type_B"] == True) &
            (adata.obs["B_follicle"] == True)
    )

    subset_idx = np.where(mask)[0]

    if len(subset_idx) == 0:
        adata.obs["follicle_cluster"] = -1
        return adata, 0

    coords = adata.obsm["spatial"][subset_idx]

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit(coords)

    labels = clustering.labels_

    valid_labels = []
    for label in np.unique(labels):
        if label == -1:
            continue

        n_cells = np.sum(labels == label)
        if n_cells >= min_cluster_size:
            valid_labels.append(label)

    final_labels = np.full(len(adata), -1)

    for i, idx in enumerate(subset_idx):
        if labels[i] in valid_labels:
            final_labels[idx] = labels[i]

    adata.obs["follicle_cluster"] = final_labels

    return adata, len(valid_labels)


def plot_follicles(adata, sample_name, out_dir):
    coords = adata.obsm["spatial"]

    plt.figure(figsize=(10, 10))
    plt.grid(False)
    plt.gca().set_aspect("equal", adjustable="box")

    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        color="lightgrey",
        s=3,
        alpha=0.4
    )

    follicle_mask = adata.obs["follicle_cluster"] >= 0

    if follicle_mask.sum() > 0:
        sns.scatterplot(
            x=coords[follicle_mask, 0],
            y=coords[follicle_mask, 1],
            hue=adata.obs.loc[follicle_mask, "follicle_cluster"].astype(str),
            palette="tab20",
            s=8,
            linewidth=0
        )

    plt.title(sample_name)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{sample_name}_follicles.png",
        dpi=300
    )

    plt.show()

def detect_follicles_hdbscan_all_cells(
    adata,
    min_cluster_size=200
):
    """
    Run HDBSCAN on ALL spatial points (no BANKSY filtering).
    Returns cluster labels for all cells.
    """

    coords = adata.obsm["spatial"]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size
    )

    labels = clusterer.fit_predict(coords)

    adata.obs["follicle_cluster_hdb"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return adata, n_clusters