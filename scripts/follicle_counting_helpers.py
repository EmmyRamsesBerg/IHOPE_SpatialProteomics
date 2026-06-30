import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

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

def normalize_follicle_counts(df, per_n_cells=1000):
    """
    Add a column with follicle counts normalized to a fixed number of cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'n_follicles' and 'total_cells' columns.
    per_n_cells : int
        The cell count to normalize to (default 1000).

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with a new column 'follicles_per_1000cells'.
    """
    df = df.copy()
    df["follicles_per_1000cells"] = (
        df["n_follicles"] / df["total_cells"] * per_n_cells
    ).round(2)
    return df



